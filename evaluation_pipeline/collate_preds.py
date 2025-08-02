from __future__ import annotations

import json
import pathlib
import argparse
import numpy as np

BLIMP_FAST_SIZE = 200
SUPPLEMENT_FAST_SIZE = 50
EWOK_FAST_SIZE = 100
ENTITY_TRACKING_FAST_SIZE = {"regular_0_ops": 606, "regular_1_ops": 607, "regular_2_ops": 606, "regular_3_ops": 606, "regular_4_ops": 568, "regular_5_ops": 159}

READING_SIZE = 1726
WUG_SIZE = 200
WUG_PAST_TENSE_SIZE = 50

WINOGROUND_SIZE = 746
VQA_SIZE = 25230
DEVBENCH = {"things": (1854, 1854), "trog": (76, 4, 1), "lex-viz_vocab": (119, 4, 1)}

BLIMP_SIZES = {
    'superlative_quantifiers_2': 986,
    'superlative_quantifiers_1': 979,
    'anaphor_gender_agreement': 971,
    'principle_A_reconstruction': 967,
    'irregular_past_participle_adjectives': 961,
    'sentential_subject_island': 961,
    'wh_island': 960,
    'left_branch_island_simple_question': 951,
    'coordinate_structure_constraint_object_extraction': 949,
    'tough_vs_raising_1': 948,
    'left_branch_island_echo_question': 947,
    'principle_A_c_command': 946,
    'regular_plural_subject_verb_agreement_2': 945,
    'irregular_past_participle_verbs': 942,
    'determiner_noun_agreement_with_adj_2': 941,
    'principle_A_domain_3': 941,
    'determiner_noun_agreement_with_adjective_1': 933,
    'anaphor_number_agreement': 931,
    'determiner_noun_agreement_2': 931,
    'existential_there_quantifiers_1': 930,
    'determiner_noun_agreement_1': 929,
    'matrix_question_npi_licensor_present': 929,
    'adjunct_island': 928,
    'existential_there_subject_raising': 924,
    'animate_subject_trans': 923,
    'drop_argument': 920,
    'tough_vs_raising_2': 920,
    'wh_vs_that_with_gap': 919,
    'sentential_negation_npi_licensor_present': 919,
    'principle_A_case_2': 915,
    'principle_A_domain_2': 915,
    'principle_A_domain_1': 914,
    'npi_present_2': 914,
    'principle_A_case_1': 912,
    'existential_there_quantifiers_2': 911,
    'wh_vs_that_with_gap_long_distance': 910,
    'npi_present_1': 909,
    'coordinate_structure_constraint_complex_left_branch': 906,
    'passive_2': 903,
    'wh_questions_subject_gap': 898,
    'animate_subject_passive': 895,
    'irregular_plural_subject_verb_agreement_2': 892,
    'regular_plural_subject_verb_agreement_1': 890,
    'only_npi_licensor_present': 882,
    'wh_vs_that_no_gap_long_distance': 875,
    'distractor_agreement_relative_clause': 871,
    'sentential_negation_npi_scope': 871,
    'intransitive': 868,
    'transitive': 868,
    'wh_vs_that_no_gap': 861,
    'wh_questions_object_gap': 859,
    'wh_questions_subject_gap_long_distance': 857,
    'inchoative': 855,
    'complex_NP_island': 846,
    'passive_1': 840,
    'determiner_noun_agreement_with_adj_irregular_2': 840,
    'only_npi_scope': 837,
    'ellipsis_n_bar_2': 828,
    'determiner_noun_agreement_irregular_2': 820,
    'causative': 818,
    'existential_there_object_raising': 812,
    'irregular_plural_subject_verb_agreement_1': 804,
    'ellipsis_n_bar_1': 802,
    'distractor_agreement_relational_noun': 788,
    'expletive_it_object_raising': 759,
    'determiner_noun_agreement_with_adj_irregular_1': 718,
    'determiner_noun_agreement_irregular_1': 681
}

SUPPLEMENT_SIZES = {"hypernym": 842, "qa_congruence_easy": 64, "qa_congruence_tricky": 165, "subject_aux_inversion": 3867, "turn_taking": 280}

EWOK_SIZES = {
    'agent-properties': 2210,
    'social-relations': 1548,
    'physical-relations': 818,
    'material-dynamics': 770,
    'physical-interactions': 556,
    'spatial-relations': 490,
    'social-properties': 328,
    'quantitative-properties': 314,
    'social-interactions': 294,
    'material-properties': 170,
    'physical-dynamics': 120
}

ENTITY_TRACKING_SIZES = {
    "regular_0_ops": 606,
    "regular_1_ops": 607,
    "regular_2_ops": 606,
    "regular_3_ops": 606,
    "regular_4_ops": 568,
    "regular_5_ops": 159,
    "ambiref_0_ops": 607,
    "ambiref_1_ops": 607,
    "ambiref_2_ops": 604,
    "ambiref_3_ops": 604,
    "ambiref_4_ops": 615,
    "ambiref_5_ops": 187,
    "move_contents_0_ops": 605,
    "move_contents_1_ops": 606,
    "move_contents_2_ops": 605,
    "move_contents_3_ops": 606,
    "move_contents_4_ops": 529,
    "move_contents_5_ops": 156
}
COMPS_SIZES = {
    "base": 49340,
    "wugs_dist_before": 13896,
    "wugs_dist_in_between": 13896,
    "wugs": 13896,
}
BOOLQ_SIZE = 1635
MNLI_SIZE = 4908
MRPC_SIZE = 204
MULTIRC_SIZE = 2424
QQP_SIZE = 20215
RTE_SIZE = 139
WSC_SIZE = 52

FAST_SIZES = {
    "ewok": EWOK_FAST_SIZE,
    "blimp": BLIMP_FAST_SIZE,
    "blimp_supplement": SUPPLEMENT_FAST_SIZE,
    "entity_tracking": ENTITY_TRACKING_FAST_SIZE,
    "reading": READING_SIZE,
    "wug_adj": WUG_SIZE,
    "wug_past": WUG_PAST_TENSE_SIZE
}
FULL_SIZES = {
    "ewok": EWOK_SIZES,
    "blimp": BLIMP_SIZES,
    "blimp_supplement": SUPPLEMENT_SIZES,
    "entity_tracking": ENTITY_TRACKING_SIZES,
    "reading": READING_SIZE,
    "wug_adj": WUG_SIZE,
    "wug_past": WUG_PAST_TENSE_SIZE,
    "comps": COMPS_SIZES,
    "boolq": BOOLQ_SIZE,
    "mnli": MNLI_SIZE,
    "mrpc": MRPC_SIZE,
    "multirc": MULTIRC_SIZE,
    "qqp": QQP_SIZE,
    "rte": RTE_SIZE,
    "wsc": WSC_SIZE,
    "winoground": WINOGROUND_SIZE,
    "vqa": VQA_SIZE
}


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_path_or_name", required=True, type=pathlib.Path, help="Name of the model to collate the results from")
    parser.add_argument("--backend", required=True, type=str, help="The backend used during evaluation", choices=["mlm", "causal", "mntp", "enc_dec_mask", "enc_dec_prefix"])

    parser.add_argument("--results_dir", default="results", type=pathlib.Path, help="Path to the results directory.")
    parser.add_argument("--revision_name", default="main", type=str, help="Name of the checkpoint/version of the model to test.")
    parser.add_argument("--fast", action="store_true", help="Whether to get the fast evaluation instead of the fast.")
    parser.add_argument("--multimodal", action="store_true", help="Whether mutlimodal evaluation was done.")

    return parser.parse_args()


def _check_validity_of_dir(args: argparse.Namespace) -> bool:
    finetune_path = args.results_dir / args.model_path_or_name.stem / args.revision_name / "finetune"
    zero_shot_path = args.results_dir / args.model_path_or_name.stem / args.revision_name / "zero_shot" / args.backend
    devbench_path = args.results_dir / args.model_path_or_name.stem / args.revision_name / "zero_shot"
    valid = True

    if args.fast:
        if not (zero_shot_path / "blimp" / "blimp_fast" / "predictions.json").exists():
            print("The blimp data is missing!")
            valid = False
        if not (zero_shot_path / "blimp" / "supplement_fast" / "predictions.json").exists():
            print("The blimp supplement data is missing!")
            valid = False
        if not (zero_shot_path / "ewok" / "ewok_fast" / "predictions.json").exists():
            print("The ewok data is missing!")
            valid = False
        if not (zero_shot_path / "entity_tracking" / "entity_tracking_fast" / "predictions.json").exists():
            print("The entity tracking data is missing!")
            valid = False
        if not (zero_shot_path / "wug_adj" / "wug_adj_nominalization" / "predictions.json").exists():
            print("The wug adj nominalization data is missing!")
            valid = False
        if not (zero_shot_path / "wug_past" / "wug_past_tense" / "predictions.json").exists():
            print("The wug past tense data is missing!")
            valid = False
        if not (zero_shot_path / "reading" / "predictions.json").exists():
            print("The reading data is missing!")
            valid = False
    else:
        if not (finetune_path / "boolq" / "predictions.json").exists():
            print("The boolq data is missing!")
            valid = False
        if not (finetune_path / "mnli" / "predictions.json").exists():
            print("The mnli data is missing!")
            valid = False
        if not (finetune_path / "mrpc" / "predictions.json").exists():
            print("The mrpc data is missing!")
            valid = False
        if not (finetune_path / "multirc" / "predictions.json").exists():
            print("The multirc data is missing!")
            valid = False
        if not (finetune_path / "qqp" / "predictions.json").exists():
            print("The qqp data is missing!")
            valid = False
        if not (finetune_path / "rte" / "predictions.json").exists():
            print("The rte data is missing!")
            valid = False
        if not (finetune_path / "wsc" / "predictions.json").exists():
            print("The wsc data is missing!")
            valid = False
        if not (zero_shot_path / "blimp" / "blimp_filtered" / "predictions.json").exists():
            print("The blimp data is missing!")
            valid = False
        if not (zero_shot_path / "blimp" / "supplement_filtered" / "predictions.json").exists():
            print("The blimp supplement data is missing!")
            valid = False
        if not (zero_shot_path / "ewok" / "ewok_filtered" / "predictions.json").exists():
            print("The ewok data is missing!")
            valid = False
        if not (zero_shot_path / "entity_tracking" / "entity_tracking" / "predictions.json").exists():
            print("The entity tracking data is missing!")
            valid = False
        if not (zero_shot_path / "wug_adj" / "wug_adj_nominalization" / "predictions.json").exists():
            print("The wug adj nominalization data is missing!")
            valid = False
        if not (zero_shot_path / "wug_past" / "wug_past_tense" / "predictions.json").exists():
            print("The wug past tense data is missing!")
            valid = False
        if not (zero_shot_path / "comps" / "comps" / "predictions.json").exists():
            print("The comps data is missing!")
            valid = False
        if not (zero_shot_path / "reading" / "predictions.json").exists():
            print("The reading data is missing!")
            valid = False
        if args.multimodal:
            if not (zero_shot_path / "vqa" / "vqa_filtered" / "predictions.json").exists():
                print("The vqa data is missing!")
                valid = False
            if not (zero_shot_path / "winoground" / "winoground_filtered" / "predictions.json").exists():
                print("The winoground data is missing!")
                valid = False
            if not (devbench_path / "devbench" / "lex-viz_vocab.npy").exists():
                print("The devbench visual vocabulary data is missing!")
                valid = False
            if not (devbench_path / "devbench" / "gram-trog.npy").exists():
                print("The devbench TROG data is missing!")
                valid = False
            if not (devbench_path / "devbench" / "sem-things_pairwise_sims.npy").exists():
                print("The devbench things data is missing!")
                valid = False

    return valid


def _check_size(task: str, results: dict[str, dict[str, list[dict[str, str | int | float]]]], fast: bool) -> bool:
    valid = True

    for key, res in results.items():
        if fast:
            if task == "entity_tracking":
                if len(res["predictions"]) != FAST_SIZES[task][key]:
                    valid = False
                    print(f"The sub-data {key} from {task} has {len(res['predictions'])} datapoints, when it should have {FAST_SIZES[task][key]} datapoints!")
            else:
                if len(res["predictions"]) != FAST_SIZES[task]:
                    valid = False
                    print(f"The sub-data {key} from {task} has {len(res['predictions'])} datapoints, when it should have {FAST_SIZES[task]} datapoints!")
        else:
            if isinstance(FULL_SIZES[task], dict):
                if len(res["predictions"]) != FULL_SIZES[task][key]:
                    valid = False
                    print(f"The sub-data {key} from {task} has {len(res['predictions'])} datapoints, when it should have {FULL_SIZES[task][key]} datapoints!")
            else:
                if len(res["predictions"]) != FULL_SIZES[task]:
                    valid = False
                    print(f"The sub-data {key} from {task} has {len(res['predictions'])} datapoints, when it should have {FULL_SIZES[task]} datapoints!")
    return valid


def _check_size_devbench(subtask: str, results: np.array[float]) -> bool:
    valid = True
    required_shape = DEVBENCH[subtask]

    if results.shape != required_shape:
        print(f"Error: Wrong shape for results for `{subtask}` in `devbench`.")
        valid = False
    if not str(results.dtype).startswith("float"):
        print(f"Error: Results for `{subtask}` (devbench) should be floats but aren't.")
        valid = False

    return valid


def _load_results(path: pathlib.Path) -> dict[str, dict[str, list[dict[str, str | int | float]]]]:
    with path.open("r") as fj:
        results: dict[str, dict[str, list[dict[str, str | int | float]]]] = json.load(fj)

    return results


def _load_results_devbench(path: pathlib.Path) -> np.array[float]:
    results = np.load(path)
    return results


def collate_preds(args: argparse.Namespace) -> None:
    assert _check_validity_of_dir(args), "The directory is incorrect, please fix the errors!"

    if args.fast:
        full_results = {}
        main_path: pathlib.Path = args.results_dir / args.model_path_or_name.stem / args.revision_name / "zero_shot" / args.backend
        output_path: pathlib.Path = args.results_dir / args.model_path_or_name.stem / args.revision_name / f"all_fast_preds_{args.backend}.json"

        # BLiMP
        blimp_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(main_path / "blimp" / "blimp_fast" / "predictions.json")
        assert _check_size("blimp", blimp_results, True), "The BLiMP Fast data is incorrect"
        full_results["blimp"] = blimp_results

        # BLiMP Supplement
        bsupp_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(main_path / "blimp" / "supplement_fast" / "predictions.json")
        assert _check_size("blimp_supplement", bsupp_results, True), "The BLiMP Supplement Fast data is incorrect"
        full_results["blimp_supplement"] = bsupp_results

        # EWoK
        ewok_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(main_path / "ewok" / "ewok_fast" / "predictions.json")
        assert _check_size("ewok", ewok_results, True), "The EWoK Fast data is incorrect"
        full_results["ewok"] = ewok_results

        # Entity Tracking
        et_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(main_path / "entity_tracking" / "entity_tracking_fast" / "predictions.json")
        assert _check_size("entity_tracking", et_results, True), "The Entity Tracking Fast data is incorrect"
        full_results["entity_tracking"] = et_results

        # WUG_ADJ
        wug_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(main_path / "wug_adj" / "wug_adj_nominalization" / "predictions.json")
        assert _check_size("wug_adj", wug_results, True), "The WUG Adjective Nominalization data is incorrect"
        full_results["wug_adj"] = wug_results

        # WUG_PAST
        wug_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(main_path / "wug_past" / "wug_past_tense" / "predictions.json")
        assert _check_size("wug_past", wug_results, True), "The WUG_PAST data is incorrect"
        full_results["wug_past"] = wug_results

        # Reading
        read_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(main_path / "reading" / "predictions.json")
        assert _check_size("reading", read_results, True), "The Reading data is incorrect"
        full_results["reading"] = read_results

        with output_path.open("w") as fj:
            json.dump(full_results, fj)
    else:
        full_results = {}
        zero_main_path: pathlib.Path = args.results_dir / args.model_path_or_name.stem / args.revision_name / "zero_shot" / args.backend
        devbench_path: pathlib.Path = args.results_dir / args.model_path_or_name.stem / args.revision_name / "zero_shot"
        fine_main_path: pathlib.Path = args.results_dir / args.model_path_or_name.stem / args.revision_name / "finetune"
        output_path: pathlib.Path = args.results_dir / args.model_path_or_name.stem / args.revision_name / f"all_full_preds_{args.backend}.json"

        # BLiMP
        blimp_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(zero_main_path / "blimp" / "blimp_filtered" / "predictions.json")
        assert _check_size("blimp", blimp_results, args.fast), "The BLiMP data is incorrect"
        full_results["blimp"] = blimp_results

        # BLiMP Supplement
        bsupp_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(zero_main_path / "blimp" / "supplement_filtered" / "predictions.json")
        assert _check_size("blimp_supplement", bsupp_results, args.fast), "The BLiMP Supplement data is incorrect"
        full_results["blimp_supplement"] = bsupp_results

        # EWoK
        ewok_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(zero_main_path / "ewok" / "ewok_filtered" / "predictions.json")
        assert _check_size("ewok", ewok_results, args.fast), "The EWoK data is incorrect"
        full_results["ewok"] = ewok_results

        # Entity Tracking
        et_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(zero_main_path / "entity_tracking" / "entity_tracking" / "predictions.json")
        assert _check_size("entity_tracking", et_results, args.fast), "The Entity Tracking data is incorrect"
        full_results["entity_tracking"] = et_results

        # WUG_ADJ
        wug_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(zero_main_path / "wug_adj" / "wug_adj_nominalization" / "predictions.json")
        assert _check_size("wug_adj", wug_results, args.fast), "The WUG Adjective Nominalization data is incorrect"
        full_results["wug_adj"] = wug_results

        # WUG_PAST
        wug_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(zero_main_path / "wug_past" / "wug_past_tense" / "predictions.json")
        assert _check_size("wug_past", wug_results, args.fast), "The WUG Past Tense data is incorrect"
        full_results["wug_past"] = wug_results

        # WUG_PAST
        comps_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(zero_main_path / "comps" / "comps" / "predictions.json")
        assert _check_size("comps", comps_results, args.fast), "The COMPS data is incorrect"
        full_results["comps"] = comps_results

        # Reading
        read_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(zero_main_path / "reading" / "predictions.json")
        assert _check_size("reading", read_results, args.fast), "The Reading data is incorrect"
        full_results["reading"] = read_results

        # GLUE
        full_results["glue"] = {}

        glue_tasks = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]

        for gt in glue_tasks:
            read_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(fine_main_path / gt / "predictions.json")
            assert _check_size(gt, read_results, args.fast), f"The {gt} data is incorrect"
            full_results["glue"] |= read_results

        if args.multimodal:
            # Multi-Modal
            # VQA
            read_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(zero_main_path / "vqa" / "vqa_filtered" / "predictions.json")
            assert _check_size("vqa", read_results, args.fast), "The VQA data is incorrect"
            full_results["vqa"] = read_results

            # Winoground
            read_results: dict[str, dict[str, list[dict[str, str | int | float]]]] = _load_results(zero_main_path / "winoground" / "winoground_filtered" / "predictions.json")
            assert _check_size("winoground", read_results, args.fast), "The Winoground data is incorrect"
            full_results["winoground"] = read_results

            # DevBench
            # Visual vocabulary
            full_results['devbench'] = {'lex-viz_visual' : {}, 'trog' : {}, 'things' : {}}

            read_results: np.array[float] = _load_results_devbench(devbench_path / "devbench" / "lex-viz_vocab.npy")
            assert _check_size_devbench("lex-viz_vocab", read_results), "The DevBench Visual vocabulary data is incorrect"
            full_results["devbench"]["lex-viz_visual"]["predictions"] = read_results.tolist()

            # TROG
            read_results: np.array[float] = _load_results_devbench(devbench_path / "devbench" / "gram-trog.npy")
            assert _check_size_devbench("trog", read_results), "The DevBench TROG data is incorrect"
            full_results["devbench"]["trog"]["predictions"] = read_results.tolist()

            # Things
            read_results: np.array[float] = _load_results_devbench(devbench_path / "devbench" / "sem-things_pairwise_sims.npy")
            assert _check_size_devbench("things", read_results), "The DevBench things data is incorrect"
            full_results["devbench"]["things"]["predictions"] = read_results.tolist()

        with output_path.open("w") as fj:
            json.dump(full_results, fj)


if __name__ == "__main__":
    args = _parse_arguments()

    collate_preds(args)
