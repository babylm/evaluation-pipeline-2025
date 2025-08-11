#!/usr/bin/env python
import json
import logging
import random
import sys
import typing as t
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
random.seed(42)
T = t.TypeVar("T")


class StepConfig:
    """Configuration for step-wise analysis of model checkpoints."""

    def __init__(
        self,
        resume: bool = False,
        track: str = "non-strict-small",
        debug: bool = False,
        file_path: Path | None = None,
    ) -> None:
        self.steps, self.word_counts = self.generate_checkpoint_steps(track)

        if debug:
            self.steps = self.steps[:5]
            self.word_counts = self.word_counts[:5]
            logger.info("Entering debugging mode, select first 5 exisitng checkpoints.")

        if resume and file_path is not None:
            self.steps, self.word_counts = self.recover_steps(file_path)

        logger.info(f"Generated {len(self.steps)} checkpoint steps")

    def generate_checkpoint_steps(self, track) -> list[int]:
        """Generate complete list of checkpoint steps."""
        M = 10**6
        checkpoint_names = [f"chck_{i}M" for i in range(1, 10)] + [
            f"chck_{i * 10}M" for i in range(1, 11)
        ]
        word_counts = [i * M for i in range(1, 10)] + [10 * i * M for i in range(1, 11)]
        if track != "strict-small":
            checkpoint_names = checkpoint_names + [
                f"chck_{i * 100}M" for i in range(2, 11)
            ]
            word_counts = word_counts + [100 * i * M for i in range(2, 11)]
        return checkpoint_names, word_counts

    def recover_steps(self, file_path: Path) -> list[int]:
        """Filter out steps that have already been processed based on JSON keys."""
        if not file_path.is_file():
            return self.steps, self.word_counts

        try:
            data = JsonProcessor.load_json(file_path)
            completed_steps = set()

            if isinstance(data, dict) and "results" in data:
                for result in data["results"]:
                    if "step" in result:
                        completed_steps.add(result["step"])
            elif isinstance(data, list):
                for result in data:
                    if isinstance(result, dict) and "step" in result:
                        completed_steps.add(result["step"])

            new_steps = [step for step in self.steps if step not in completed_steps]
            new_word_counts = self.word_counts[-len(new_steps) :]
            return new_steps, new_word_counts
        except Exception as e:
            logger.warning(f"Error reading resume file: {e}")
            return self.steps, self.word_counts


class JsonProcessor:
    """Class for handling JSON serialization with NumPy type conversion."""

    @staticmethod
    def convert_numpy_types(obj: t.Any, _seen: set[int] | None = None) -> t.Any:
        """Recursively convert NumPy types and custom objects in a nested structure to standard Python types."""
        if _seen is None:
            _seen = set()

        obj_id = id(obj)
        if obj_id in _seen:
            return f"<circular_reference_to_{type(obj).__name__}>"

        if obj is None:
            return None

        if (
            hasattr(obj, "__module__")
            and obj.__module__
            and "networkx" in obj.__module__
        ):
            class_name = obj.__class__.__name__
            if "Graph" in class_name:
                return {
                    "type": "networkx_graph",
                    "graph_type": class_name,
                    "nodes": list(obj.nodes()) if hasattr(obj, "nodes") else [],
                    "edges": list(obj.edges()) if hasattr(obj, "edges") else [],
                    "number_of_nodes": obj.number_of_nodes()
                    if hasattr(obj, "number_of_nodes")
                    else 0,
                    "number_of_edges": obj.number_of_edges()
                    if hasattr(obj, "number_of_edges")
                    else 0,
                }
            return f"<networkx_{class_name}>"

        if hasattr(obj, "__class__") and obj.__class__.__name__ == "SearchResult":
            _seen.add(obj_id)
            try:
                result_dict = {
                    "neurons": JsonProcessor.convert_numpy_types(obj.neurons, _seen),
                    "delta_loss": JsonProcessor.convert_numpy_types(
                        obj.delta_loss, _seen
                    ),
                }
                if hasattr(obj, "is_target_size"):
                    result_dict["is_target_size"] = obj.is_target_size
                return result_dict
            finally:
                _seen.discard(obj_id)

        if isinstance(obj, np.ndarray):
            return JsonProcessor.convert_numpy_types(obj.tolist(), _seen)

        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(
            obj,
            (
                np.integer,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.complex128):
            return complex(obj)

        if isinstance(obj, Path):
            return str(obj)

        if hasattr(obj, "__class__"):
            class_name = obj.__class__.__name__
            if class_name in ["Graph", "DiGraph", "MultiGraph", "MultiDiGraph"]:
                return {
                    "type": "networkx_graph",
                    "graph_type": class_name,
                    "nodes": list(obj.nodes()),
                    "edges": list(obj.edges(data=True))
                    if hasattr(obj, "edges")
                    else [],
                    "number_of_nodes": obj.number_of_nodes()
                    if hasattr(obj, "number_of_nodes")
                    else 0,
                    "number_of_edges": obj.number_of_edges()
                    if hasattr(obj, "number_of_edges")
                    else 0,
                }
            if class_name in [
                "AdjacencyView",
                "AtlasView",
                "NodeView",
                "EdgeView",
                "DegreeView",
            ]:
                try:
                    return list(obj) if hasattr(obj, "__iter__") else str(obj)
                except:
                    return f"<{class_name}_object>"

        _seen.add(obj_id)

        try:
            if isinstance(obj, dict):
                return {
                    JsonProcessor.convert_numpy_types(
                        k, _seen
                    ): JsonProcessor.convert_numpy_types(v, _seen)
                    for k, v in obj.items()
                }

            if isinstance(obj, (list, tuple)):
                converted = [
                    JsonProcessor.convert_numpy_types(item, _seen) for item in obj
                ]
                return converted if isinstance(obj, list) else tuple(converted)

            if isinstance(obj, set):
                return [JsonProcessor.convert_numpy_types(item, _seen) for item in obj]

            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return JsonProcessor.convert_numpy_types(obj.to_dict(), _seen)

            if hasattr(obj, "__dict__") and not isinstance(obj, type):
                class_name = obj.__class__.__name__

                if any(
                    nx_type in class_name
                    for nx_type in ["Graph", "View", "Atlas", "Node", "Edge", "Degree"]
                ):
                    return f"<{class_name}_skipped>"

                if class_name in [
                    "ValidationResult",
                    "StatisticalTest",
                    "StatisticalValidator",
                    "BootstrapEstimator",
                ]:
                    safe_dict = {}
                    for key, value in obj.__dict__.items():
                        if not key.startswith("_") and key not in [
                            "graph",
                            "data",
                            "samples",
                            "_seen",
                        ]:
                            try:
                                safe_dict[key] = JsonProcessor.convert_numpy_types(
                                    value, _seen
                                )
                            except (
                                RecursionError,
                                TypeError,
                                AttributeError,
                                ValueError,
                            ):
                                if isinstance(
                                    value, (int, float, str, bool, type(None))
                                ):
                                    safe_dict[key] = value
                                else:
                                    safe_dict[key] = str(type(value).__name__)
                    return safe_dict
                return JsonProcessor.convert_numpy_types(obj.__dict__, _seen)

            return obj

        finally:
            _seen.discard(obj_id)

    @classmethod
    def save_json(cls, data: dict, filepath: Path) -> None:
        """Save a nested dictionary with float values to a file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        converted_data = cls.convert_numpy_types(data)
        with open(filepath, "w") as f:
            json.dump(converted_data, f, indent=2)

    @staticmethod
    def load_json(filepath: Path) -> dict:
        """Load a JSON file into a dictionary."""
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)


class StepPathProcessor:
    """Process paths and manage steps for resumable processing."""

    def __init__(self, abl_path: Path):
        self.abl_path = abl_path
        self.step_dirs: list[tuple[Path, int]] = []

    def sort_paths(self) -> list[tuple[Path, int]]:
        """Get the sorted directory by steps in descending order."""
        step_dirs = []
        for step in self.abl_path.iterdir():
            if step.is_dir():
                try:
                    step_num = int(step.name)
                    step_dirs.append((step, step_num))
                except:
                    logger.info(f"Something wrong with step {step}")

        step_dirs.sort(key=lambda x: x[1], reverse=True)
        self.step_dirs = step_dirs
        return self.step_dirs

    def resume_results(
        self, resume: bool, save_path: Path, file_path: Path = None
    ) -> tuple[dict, list[tuple[Path, int]]]:
        """Resume results from the existing directory list."""
        if not self.step_dirs:
            self.sort_paths()

        if resume and save_path.is_file():
            final_results, remaining_step_dirs = self._get_step_intersection(
                save_path, self.step_dirs
            )

            if file_path and file_path.is_file():
                logger.info(
                    f"Filter steps from existing neuron index file. Steps before filtering: {len(remaining_step_dirs)}"
                )
                _, remaining_step_dirs = self._get_step_intersection(
                    file_path, remaining_step_dirs
                )
                logger.info(f"Steps after filtering: {len(remaining_step_dirs)}")

            logger.info(
                f"Resume {len(self.step_dirs) - len(remaining_step_dirs)} states from {save_path}."
            )

            if len(remaining_step_dirs) == 0:
                logger.info("All steps already processed. Exiting.")
                sys.exit(0)

            return final_results, remaining_step_dirs

        return {}, self.step_dirs

    def _get_step_intersection(
        self, file_path: Path, remaining_step_dirs: list[tuple[Path, int]]
    ) -> list[tuple[Path, int]]:
        """Resume results from the selected indices."""
        final_results = JsonProcessor.load_json(file_path)
        completed_results = list(final_results.keys())
        remaining_step_dirs = [
            p for p in self.step_dirs if p[0].name not in completed_results
        ]
        return final_results, remaining_step_dirs


def load_eval(
    word_path: Path, min_context: int = 20, debug: bool = False
) -> tuple[list[str], list[list[str]]]:
    """Load word and context lists from a JSON file."""
    data = JsonProcessor.load_json(word_path)
    target_words = list(data.keys())
    words = []
    contexts = []

    for word in target_words:
        if len(word) > 1:
            word_contexts = []
            word_data = data[word]
            if len(word_data) >= min_context:
                words.append(word)
                for context_data in word_data:
                    word_contexts.append(context_data["context"])
                contexts.append(word_contexts)

    if debug:
        target_words, contexts = target_words[:5], contexts[:5]
        logger.info("Entering debugging mode. Loading first 5 words")

    if not debug:
        logger.info(f"{len(target_words) - len(words)} words are filtered.")
        logger.info(f"Loading {len(words)} words.")

    return words, contexts
