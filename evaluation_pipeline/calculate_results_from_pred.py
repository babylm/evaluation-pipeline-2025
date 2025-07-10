import json
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
import argparse


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_path_or_name", required=True, type=Path, help="Name of the model to collate the results from")

    parser.add_argument("--results_dir", default="results", type=Path, help="Path to the results directory.")
    parser.add_argument("--evaluation_data_dir", default="evaluation_data", type=Path, help="Path to the evaluation data directory.")
    parser.add_argument("--revision_name", default="main", type=str, help="Name of the checkpoint/version of the model to test.")
    parser.add_argument("--fast", action="store_true", help="Whether to get the fast evaluation instead of the fast.")
    parser.add_argument("--multimodal", action="store_true", help="Whether mutlimodal evaluation was done.")

    return parser.parse_args()


def _calculate_blimp_results(results_dict: dict[str, dict[str, list[dict[str, str]]]], path_to_data: Path) -> float:
    scores = []
    for subtask in results_dict.keys():
        correct = 0
        total = 0
        with (path_to_data / subtask).with_suffix(".jsonl").open("r") as data_file:
            subtask_results = results_dict[subtask]["predictions"]
            for result, data in zip(subtask_results, data_file.readlines()):
                data = json.loads(data)
                total += 1
                if result["pred"].strip() == data["sentence_good"].strip():
                    correct += 1
            scores.append((correct / total) * 100)
    return sum(scores) / len(scores)


def _calculate_ewok_results(results_dict: dict[str, dict[str, list[dict[str, str]]]], path_to_data: Path) -> float:
    scores = []
    for subtask in results_dict.keys():
        correct = 0
        total = 0
        with (path_to_data / subtask).with_suffix(".jsonl").open("r") as data_file:
            subtask_results = results_dict[subtask]["predictions"]
            for result, data in zip(subtask_results, data_file.readlines()):
                data = json.loads(data)
                # print(result["pred"], data["Target1"])
                total += 1
                if result["pred"].strip() == data["Target1"].strip():
                    correct += 1
            scores.append((correct / total) * 100)
    return sum(scores) / len(scores)


def _calculate_wug_results(results_dict: dict[str, dict[str, list[dict[str, str]]]], path_to_data: Path) -> float:
    scores = []
    for subtask in results_dict.keys():
        correct = 0
        total = 0
        with (path_to_data / subtask).with_suffix(".jsonl").open("r") as data_file:
            subtask_results = results_dict[subtask]["predictions"]
            for result, data in zip(subtask_results, data_file.readlines()):
                data = json.loads(data)
                sentences = data["sentences"].split("\t")
                total += 1
                if result["pred"].strip() == sentences[0].strip():
                    correct += 1
            scores.append((correct / total) * 100)
    return sum(scores) / len(scores)


def _calculate_entity_tracking_results(results_dict: dict[str, dict[str, list[dict[str, str]]]], path_to_data: Path) -> float:
    scores = []
    for subtask in path_to_data.glob("**/*"):
        correct = 0
        total = 0
        if subtask.suffix == ".jsonl":
            with subtask.open("r") as data_file:
                num_ops = 0
                idx = 0
                subsubtask = "_".join([subtask.stem, str(num_ops), "ops"])
                subsubtask_results = results_dict[subsubtask]["predictions"]
                for data in data_file.readlines():
                    data = json.loads(data)
                    if data["numops"] != num_ops:
                        scores.append((correct / total) * 100)
                        correct = 0
                        total = 0
                        num_ops += 1
                        idx = 0
                        subsubtask = "_".join([subtask.stem, str(num_ops), "ops"])
                        subsubtask_results = results_dict[subsubtask]["predictions"]
                    total += 1
                    if subsubtask_results[idx]["pred"].strip() == data["options"][0].strip():
                        correct += 1
                    idx += 1
                scores.append((correct / total) * 100)
    return sum(scores) / len(scores)


def _calculate_reading_results(results_dict: dict[str, dict[str, list[dict[str, int | float]]]], path_to_data: Path) -> tuple[float, float]:
    data = pd.read_csv(path_to_data, dtype={'item': str})
    preds = [item["pred"] for item in results_dict["reading"]["predictions"]]
    prev_preds = [item["prev_pred"] for item in results_dict["reading"]["predictions"]]
    data["pred"] = preds
    data["prev_pred"] = prev_preds
    eye_tracking_vars = ['RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound']
    eye_tracking_result = []
    for dv in eye_tracking_vars:
        # Baseline model
        temp = data[[dv, "Subtlex_log10", "length", "context_length"]].dropna()
        OLS_baseline = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length', data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        # Predictive model
        temp = data[[dv, "Subtlex_log10", "length", "context_length", "pred"]].dropna()
        OLS_model = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + pred', data=temp).fit()
        R2_model = float(OLS_model.rsquared)
        eye_tracking_result.append(((R2_model-R2_baseline)/(1-R2_baseline)) * 100)
    eye_tracking_result = sum(eye_tracking_result) / len(eye_tracking_result)

    # Baseline model
    temp = data[["self_paced_reading_time", "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
    OLS_baseline = smf.ols(formula='self_paced_reading_time ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred', data=temp).fit()
    R2_baseline = float(OLS_baseline.rsquared)
    # Predictive model
    temp = data[["self_paced_reading_time", "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred", "pred"]].dropna()
    OLS_model = smf.ols(formula='self_paced_reading_time ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred + pred', data=temp).fit()
    R2_model = float(OLS_model.rsquared)
    self_paced_reading_result = ((R2_model-R2_baseline)/(1-R2_baseline)) * 100

    return self_paced_reading_result, eye_tracking_result


def _calculate_glue_results(results_dict: dict[str, dict[str, list[dict[str, int]]]], path_to_data: Path) -> float:
    scores = []
    for subtask in results_dict.keys():
        correct = 0
        total = 0
        with (path_to_data / subtask).with_suffix(".valid.jsonl").open("r") as data_file:
            subtask_results = results_dict[subtask]["predictions"]
            for result, data in zip(subtask_results, data_file.readlines()):
                data = json.loads(data)
                total += 1
                if result["pred"] == data["label"]:
                    correct += 1
            scores.append((correct / total) * 100)
    return sum(scores) / len(scores)


if __name__ == "__main__":
    args = _parse_arguments()
    path_to_results = args.results_dir / args.model_path_or_name.stem / args.revision_name
    if args.fast:
        path_to_results = path_to_results / "all_fast_preds.json"
        all_results = json.load(path_to_results.open("r"))
        evaluation_path = args.evaluation_data_dir / "fast_eval"
        print("###### RESULTS ######")
        print("=====================")
        # BLiMP
        score = _calculate_blimp_results(all_results["blimp"], evaluation_path / "blimp_fast")
        print(f"BLIMP\t\t{score:.2f}")
        # Supplement
        score = _calculate_blimp_results(all_results["blimp_supplement"], evaluation_path / "supplement_fast")
        print(f"SUPPLEMENT\t{score:.2f}")
        # EWoK
        score = _calculate_ewok_results(all_results["ewok"], evaluation_path / "ewok_fast")
        print(f"EWOK\t\t{score:.2f}")
        # Entity Tracking
        score = _calculate_entity_tracking_results(all_results["entity_tracking"], evaluation_path / "entity_tracking_fast")
        print(f"ENTITY TRACKING\t{score:.2f}")
        # WUG
        score = _calculate_wug_results(all_results["wug"], evaluation_path / "wug_adj_nominalization")
        print(f"WUG\t\t{score:.2f}")
        # Reading (SPR)
        scores = _calculate_reading_results(all_results["reading"], evaluation_path / "reading" / "reading_data.csv")
        print(f"READING (SPR)\t{scores[0]:.2f}")
        print(f"READING (ET)\t{scores[1]:.2f}")
    else:
        path_to_results = path_to_results / "all_full_preds.json"
        all_results = json.load(path_to_results.open("r"))
        evaluation_path = args.evaluation_data_dir / "full_eval"
        print("###### RESULTS ######")
        print("=====================")
        # BLiMP
        score = _calculate_blimp_results(all_results["blimp"], evaluation_path / "blimp_filtered")
        print(f"BLIMP\t\t{score:.2f}")
        # Supplement
        score = _calculate_blimp_results(all_results["blimp_supplement"], evaluation_path / "supplement_filtered")
        print(f"SUPPLEMENT\t{score:.2f}")
        # EWoK
        score = _calculate_ewok_results(all_results["ewok"], evaluation_path / "ewok_filtered")
        print(f"EWOK\t\t{score:.2f}")
        # Entity Tracking
        score = _calculate_entity_tracking_results(all_results["entity_tracking"], evaluation_path / "entity_tracking")
        print(f"ENTITY TRACKING\t{score:.2f}")
        # WUG
        score = _calculate_wug_results(all_results["wug"], evaluation_path / "wug_adj_nominalization")
        print(f"WUG\t\t{score:.2f}")
        # Reading (SPR)
        scores = _calculate_reading_results(all_results["reading"], evaluation_path / "reading" / "reading_data.csv")
        print(f"READING (SPR)\t{scores[0]:.2f}")
        print(f"READING (ET)\t{scores[1]:.2f}")
        # GLUE
        score = _calculate_glue_results(all_results["glue"], evaluation_path / "glue_filtered")
        print(f"GLUE\t\t{score:.2f}")
