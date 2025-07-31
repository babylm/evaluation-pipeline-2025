import numpy as np
import pandas as pd
from evaluation_pipeline.devbench.comparison.stats_helper import get_opt_kl

VV_DIR = "evaluation_data/full_eval/devbench/evals/lex-viz_vocab/"
human_data_vv = pd.read_csv(VV_DIR + "human.csv")


def compare_vv(model_data, human_data):
    kl_values = []
    beta_values = []
    iterations = []

    grouped = human_data.groupby('age_bin')
    for _, group in grouped:
        relevant_data = group.filter(regex='image|trial')
        opt_kl = get_opt_kl(relevant_data, model_data)
        kl_values.append(opt_kl['objective'])
        beta_values.append(opt_kl['solution'])
        iterations.append(opt_kl['iterations'])

    result_df = pd.DataFrame({
        'age_bin': grouped.groups.keys(),
        'kl': kl_values,
        'beta': beta_values,
        'iterations': iterations
    })
    return result_df


def get_scores(vv_file):
    model_name, revision = vv_file.split("/")[1:3]
    res = np.load(vv_file)
    res = pd.DataFrame(res.squeeze(),
                       columns=["image1", "image2", "image3", "image4"])
    res['trial'] = np.arange(res.shape[0])+1
    res['correct'] = (res['image1'] > res['image2']) & \
                     (res['image1'] > res['image3']) & \
                     (res['image1'] > res['image4'])
    acc = res['correct'].mean()
    kls = compare_vv(res, human_data_vv)
    kls['model'] = model_name
    kls['accuracy'] = acc
    other_res_vv = pd.concat([kls]).sort_values(["model", "age_bin"]).reset_index(drop=True)
    other_res_vv.to_csv(f"results/{model_name}/{revision}/zero_shot/devbench/lex-vv_scores.csv")
    return kls
