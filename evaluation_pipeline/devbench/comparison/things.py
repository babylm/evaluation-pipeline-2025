import os
import csv
from glob import glob
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from devbench.comparison.stats_helper import *

THINGS_DIR = "evaluation_data/full_eval/devbench/evals/sem-things/"
human_data_things = scipy.io.loadmat(THINGS_DIR + "spose_similarity.mat")
human_data_things = human_data_things["spose_sim"]

def get_scores(things_file):
    model_name = os.path.splitext(os.path.basename(things_file))[0]
    res = np.load(things_file)
    res_mat = cosine_similarity(res)
    sim_file = things_file.replace(".npy", "_pairwise_sims.npy")
    np.save(sim_file, res_mat)
    sim = rsa(human_data_things, res_mat)
    # Save pairwise similarities for uploading into predictions file later

    with open(f"results/devbench/{model_name}/sem-things_scores.csv", 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["model", "rsa"])
        writer.writerow([model_name, sim])
    
    return sim
