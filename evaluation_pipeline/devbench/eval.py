import evaluation_pipeline.devbench.data_handling as data_handling
from evaluation_pipeline.devbench.comparison.things import get_scores as get_things_scores
from evaluation_pipeline.devbench.comparison.trog import get_scores as get_trog_scores
from evaluation_pipeline.devbench.comparison.viz_vocab import get_scores as get_viz_vocab_scores
import numpy as np
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model on DevBench tasks.')
    parser.add_argument('--model', type=str, help='Model name or path.')
    parser.add_argument('--revision', type=str, help="Which revision to use, if None, main is used.")
    parser.add_argument('--model_type', type=str, help='Model type.')
    parser.add_argument('--image_model', type=str, default=None, help='Image model, if separate from the main model name.')
    args = parser.parse_args()

    if args.revision is None:
        args.revision = "main"

    image_model = args.image_model if args.image_model is not None else args.model

    if args.model_type == "clip":
        from evaluation_pipeline.devbench.model_classes.clip import ClipEvalModel
        from transformers import CLIPProcessor, CLIPModel

        eval_model = ClipEvalModel(
            # tested with model = "openai/clip-vit-base-patch32"
            model=CLIPModel.from_pretrained(args.model),
            processor=CLIPProcessor.from_pretrained(args.model)
        )

    elif args.model_type == "blip":
        from evaluation_pipeline.devbench.model_classes.blip import BlipEvalModel
        from transformers import AutoProcessor, BlipForImageTextRetrieval, BlipModel

        eval_model = BlipEvalModel(
            # tested with model = "Salesforce/blip-itm-base-coco", image_model = "Salesforce/blip-image-captioning-base"
            model=BlipForImageTextRetrieval.from_pretrained(args.model),
            image_model=BlipModel.from_pretrained(image_model),
            processor=AutoProcessor.from_pretrained(args.model)
        )

    elif args.model_type == "flava":
        from evaluation_pipeline.devbench.model_classes.flava import FlavaEvalModel
        from transformers import FlavaProcessor, FlavaFeatureExtractor, FlavaForPreTraining, FlavaModel

        eval_model = FlavaEvalModel(
            # tested with model = "facebook/flava-full", image_model == model
            model=FlavaForPreTraining.from_pretrained(args.model),
            processor=FlavaProcessor.from_pretrained(args.model),
            image_model=FlavaModel.from_pretrained(image_model),
            feature_extractor=FlavaFeatureExtractor.from_pretrained(args.model)
        )

    elif args.model_type == "bridgetower":
        from evaluation_pipeline.devbench.model_classes.bridgetower import BridgetowerEvalModel
        from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval, BridgeTowerModel

        eval_model = BridgetowerEvalModel(
            # tested with model = "BridgeTower/bridgetower-base-itm-mlm", image_model = "BridgeTower/bridgetower-base"
            model=BridgeTowerForImageAndTextRetrieval.from_pretrained(args.model),
            processor=BridgeTowerProcessor.from_pretrained(args.model),
            image_processor=BridgeTowerProcessor.from_pretrained(image_model),
            image_model=BridgeTowerModel.from_pretrained(image_model)
        )

    elif args.model_type == "vilt":
        from evaluation_pipeline.devbench.model_classes.vilt import ViltEvalModel
        from transformers import ViltProcessor, ViltForImageAndTextRetrieval, ViltModel

        eval_model = ViltEvalModel(
            # tested with model = "dandelin/vilt-b32-finetuned-coco", image_model = "dandelin/vilt-b32-mlm"
            model=ViltForImageAndTextRetrieval.from_pretrained(args.model),
            processor=ViltProcessor.from_pretrained(args.model),
            vilt_base_processor=ViltProcessor.from_pretrained(image_model),
            vilt_base_model=ViltModel.from_pretrained(image_model)
        )

    elif args.model_type == "cvcl":
        from evaluation_pipeline.devbench.model_classes.cvcl import CvclEvalModel
        from multimodal.multimodal_lit import MultiModalLitModel

        # tested with model = "cvcl"
        cvcl, preprocess = MultiModalLitModel.load_model(model_name=args.model)
        cvcl.eval()

        eval_model = CvclEvalModel(
            model=cvcl,
            processor=preprocess
        )

    elif args.model_type == "siglip":
        from evaluation_pipeline.devbench.model_classes.siglip import SiglipEvalModel
        from transformers import AutoProcessor, AutoModel, AutoTokenizer

        # tested with model = "google/siglip-so400m-patch14-384", image_model = "google/siglip-base-patch16-224"
        eval_model = SiglipEvalModel(
            model=AutoModel.from_pretrained(args.model),
            processor=AutoProcessor.from_pretrained(args.model),
            model_embed=AutoModel.from_pretrained(image_model),
            tokenizer=AutoTokenizer.from_pretrained(image_model),
            processor_embed=AutoProcessor.from_pretrained(image_model)
        )

    elif args.model_type == "llava":
        from evaluation_pipeline.devbench.model_classes.llava import LlavaEvalModel
        from transformers import AutoProcessor, AutoModelForPreTraining
        # tested with model = "llava-hf/llava-v1.6-mistral-7b-hf" and "bczhou/tiny-llava-v1-hf"
        is_tiny = False
        if "tiny-" in args.model:
            is_tiny = True
        eval_model = LlavaEvalModel(
                processor=AutoProcessor.from_pretrained(args.model),
                model=AutoModelForPreTraining.from_pretrained(args.model),
                is_tiny=is_tiny
        )

    elif args.model_type in ["git", "flamingo"]:
        from evaluation_pipeline.devbench.model_classes.babylm_models import BabyLMEvalModel
        from transformers import AutoProcessor, AutoModelForCausalLM

        eval_model = BabyLMEvalModel(
            model=AutoModelForCausalLM.from_pretrained(
                args.model, trust_remote_code=True),
            processor=AutoProcessor.from_pretrained(args.model,
                                                    trust_remote_code=True),
        )

    model_name = args.model.split("/")[-1]
    if not os.path.exists(f"results/{model_name}/{args.revision}/zero_shot/devbench"):
        os.makedirs(f"results/{model_name}/{args.revision}/zero_shot/devbench/")

    # Visual vocabulary
    vv_ds = data_handling.DevBenchDataset("evaluation_data/full_eval/devbench/assets/lex-viz_vocab/")
    vv_dl = data_handling.make_dataloader(vv_ds)
    vv_sims = eval_model.get_all_sim_scores(vv_dl)
    vv_file = f"results/{model_name}/{args.revision}/zero_shot/devbench/lex-viz_vocab.npy"
    np.save(vv_file, vv_sims)
    vv_score = get_viz_vocab_scores(vv_file)

    # TROG
    trog_ds = data_handling.DevBenchDataset("evaluation_data/full_eval/devbench/assets/gram-trog/")
    trog_dl = data_handling.make_dataloader(trog_ds)
    trog_sims = eval_model.get_all_sim_scores(trog_dl)
    trog_file = f"results/{model_name}/{args.revision}/zero_shot/devbench/gram-trog.npy"
    np.save(trog_file, trog_sims)
    trog_score = get_trog_scores(trog_file)

    # THINGS
    things_ds = data_handling.DevBenchDataset("evaluation_data/full_eval/devbench/assets/sem-things/")
    things_dl = data_handling.make_dataloader(things_ds)
    things_embeds = eval_model.get_all_image_feats(things_dl)
    things_file = f"results/{model_name}/{args.revision}/zero_shot/devbench/sem-things.npy"
    np.save(f"results/{model_name}/{args.revision}/zero_shot/devbench/sem-things.npy", things_embeds)
    things_score = get_things_scores(things_file)

    results = f"Visual Vocabulary Accuracy: {vv_score['accuracy']}\nTROG Accuracy: {trog_score['accuracy']}\nThings Spearman Correlation: {things_score}"
    with open(f"results/{model_name}/{args.revision}/zero_shot/devbench/results.txt", "w") as f:
        f.write(results)
