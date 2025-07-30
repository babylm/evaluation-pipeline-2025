import nlopt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# Softmax function to apply to the image columns
def softmax_images(data, beta=1):
    image_columns = data.filter(like='image')
    softmaxed_images = np.exp(beta * (image_columns - np.max(image_columns)))
    rowsum = softmaxed_images.sum(axis=1)
    softmaxed_images = softmaxed_images.div(rowsum, axis=0)
    
    data[image_columns.columns] = softmaxed_images
    return data

# Function to compute the KL divergence between two distributions
def kl_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))

# Function to calculate the mean KL divergence
def get_mean_kl_img(human_probs_wide, model_probs_wide, return_distribs=False):
    combined_distribs = pd.concat([human_probs_wide, model_probs_wide])
    combined_distribs.fillna(0, inplace=True)
    
    grouped = combined_distribs.groupby('trial')
    kl_list = []
    
    for _, distribs in grouped:
        if len(distribs) == 2:
            p = distribs.filter(like='image').iloc[0].to_numpy()
            q = distribs.filter(like='image').iloc[1].to_numpy()
            kl = kl_divergence(p, q)
            kl_list.append(kl)
    
    if return_distribs:
        return combined_distribs
    
    return np.mean(kl_list)

# Optimization function using nlopt
def get_opt_kl(human_probs_wide, model_logits_wide):
    def mean_kl(beta, _):
        model_probs_wide = softmax_images(model_logits_wide.copy(), beta[0])
        return get_mean_kl_img(human_probs_wide, model_probs_wide)
    
    opt = nlopt.opt(nlopt.GN_DIRECT_L, 1)  # Use the GN_DIRECT_L algorithm with 1 dimension (beta)
    opt.set_lower_bounds([0.025])
    opt.set_upper_bounds([40])
    opt.set_min_objective(mean_kl)
    opt.set_ftol_abs(1e-4)
    opt.set_maxeval(200)
    
    x0 = [1.0]  # Initial guess for beta
    solution = opt.optimize(x0)[0]
    objective = opt.last_optimum_value()
    iterations = opt.get_numevals()
    
    return {
        'objective': objective,
        'solution': solution,
        'iterations': iterations
    }

# Representational similarity analysis
def rsa(mat1, mat2, method="spearman"):
    # Extract the lower triangular part of the matrices
    mat1_lower = mat1[np.tril_indices_from(mat1, k=-1)]
    mat2_lower = mat2[np.tril_indices_from(mat2, k=-1)]
    
    # Compute the correlation
    if method == "spearman":
        correlation, _ = spearmanr(mat1_lower, mat2_lower)
    elif method == "pearson":
        correlation, _ = pearsonr(mat1_lower, mat2_lower)
    else:
        raise ValueError("Unsupported method. Use 'spearman' or 'pearson'.")
    
    return correlation
