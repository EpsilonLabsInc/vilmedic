#!/usr/bin/env python3
import sys
import json
import logging
import os
import glob
import re
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score

# Import your scoring classes. (These should be available in your package.)
# For example:
from f1chexbert import F1CheXbert
from radgraph import F1RadGraph

# Import additional scorers if needed
# (Assuming they are imported via relative import in your package)
from . import *
from .utils import get_logger_directory

# RadGraph package overrides logger, so we reset it:
logging.setLoggerClass(logging.Logger)

# This dictionary is not used in the code below but is kept from your original code.
REWARD_COMPLIANT = {
    "rougel": [RougeL, 1],
    "rouge2": [Rouge2, 1],
    "rouge1": [Rouge1, 1],
    "bleu": [Bleu, 1],
    "meteor": [Meteor, 1],
    "ciderdrl": [CiderDRL, 1],
    "radentitymatchexact": [RadEntityMatchExact, 1],
    "radentitynli": [RadEntityNLI, 1],
    "chexbert": [F1CheXbert, 1],
    "radgraph": [F1RadGraph, 1],
    "bertscore": [BertScore, 1],
}


def compute_scores(
    metrics,
    refs,
    hyps,
    split=0.999,
    seed=999,
    config=None,
    epoch=999,
    logger=None,
    dump=True,
    description="",
    label=""
):
    scores = dict()
    if metrics is None or not metrics:
        return scores

    assert refs is not None and hyps is not None, "You specified metrics but your evaluation does not return hyps nor refs"
    assert len(refs) == len(hyps), "refs and hyps must have same length : {} vs {}".format(len(refs), len(hyps))

    # Determine the base directory for logs
    if label:
        base_dir = f"logs/{description}/{label}"
    else:
        base_dir = f"logs/{description}"
    print(f"Creating directory: {base_dir}")
    if dump:
        os.makedirs(base_dir, exist_ok=True)

    base = f"{base_dir}/{split}_{seed}_{{}}"
    refs_file = base.format("refs.txt")
    hyps_file = base.format("hyps.txt")
    metrics_file = base.format("metrics.jsonl")

    if dump:
        with open(refs_file, "w") as f:
            f.write("\n".join(map(str, refs)))
        with open(hyps_file, "w") as f:
            f.write("\n".join(map(str, hyps)))

    for metric in metrics:
        print("Checking metric:", metric)
        if metric == "BLEU":
            scores["BLEU-1"] = Bleu(1)(refs, hyps)[0]
            scores["BLEU-2"] = Bleu(2)(refs, hyps)[0]
            scores["BLEU-3"] = Bleu(3)(refs, hyps)[0]
            scores["BLEU-4"] = Bleu(4)(refs, hyps)[0]
        elif metric == "METEOR":
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.translate.meteor_score import meteor_score

            nltk.download("wordnet")
            nltk.download("punkt")
            tokenized_references = [word_tokenize(ref) for ref in refs]
            tokenized_hypotheses = [word_tokenize(hyp) for hyp in hyps]
            meteo_scores = [
                meteor_score([ref], hyp) for ref, hyp in zip(tokenized_references, tokenized_hypotheses)
            ]
            scores["METEOR"] = np.mean(meteo_scores)
        elif metric == "CIDERD":
            scores["CIDERD"] = CiderD()(refs, hyps)[0]
        elif metric == "bertscore":
            scores["bertscore"] = BertScore()(refs, hyps)[0]
        elif metric in ["ROUGE1", "ROUGE2", "ROUGEL"]:
            scores[metric] = Rouge(rouges=[metric.lower()])(refs, hyps)[0]
        elif metric == "accuracy":
            scores["accuracy"] = round(np.mean(np.array(refs) == np.argmax(hyps, axis=-1)) * 100, 2)
        elif metric == "f1-score":
            scores["f1-score"] = classification_report(refs, np.argmax(hyps, axis=-1))
        elif metric == "auroc":
            scores["auroc"] = roc_auc_score(
                refs,
                F.softmax(torch.from_numpy(hyps), dim=-1).numpy(),
                multi_class="ovr",
            )
        elif metric == "chexbert":
            accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = F1CheXbert(
                refs_filename=base.format("refs.chexbert.txt") if dump else None,
                hyps_filename=base.format("hyps.chexbert.txt") if dump else None,
            )(hyps, refs)
            # scores["chexbert-5_micro avg_f1-score"] = chexbert_5["micro avg"]["f1-score"]
            scores["chexbert-all_micro avg_f1-score"] = chexbert_all["micro avg"]["f1-score"]
            # scores["chexbert-5_macro avg_f1-score"] = chexbert_5["macro avg"]["f1-score"]
            scores["chexbert-all_macro avg_f1-score"] = chexbert_all["macro avg"]["f1-score"]
            scores["chexbert-all"] = chexbert_all
        elif metric == "radentitymatchexact":
            scores["radentitymatchexact"] = RadEntityMatchExact()(refs, hyps)[0]
        elif metric == "radentitynli":
            scores["radentitynli"] = RadEntityNLI()(refs, hyps)[0]
        elif metric == "radgraph":
            (scores["radgraph_simple"],
             scores["radgraph_partial"],
             scores["radgraph_complete"]) = F1RadGraph(reward_level="all", model_type="radgraph-xl")(refs=refs, hyps=hyps)[0]
        elif metric == "stanford_ct_abd_accuracy":
            scores["stanford_ct_abd"] = StanfordCTAbdAcc()(refs=refs, hyps=hyps)[0]
        else:
            if logger:
                logger.warning("Metric not implemented: {}".format(metric))
            else:
                print("Metric not implemented:", metric)

    if dump:
        with open(metrics_file, "a+") as f:
            data = {"split": split, "epoch": epoch, "scores": scores, "data_size": len(refs)}
            f.write(json.dumps(data) + "\n")
    return scores


def get_text(text):
    """Extract the text following 'Assistant: ' if present."""
    assistant_key = "Assistant: "
    start_idx = text.find(assistant_key)
    if start_idx != -1:
        text = text[start_idx + len(assistant_key):]
    return text


def process_checkpoint_normal(compute_scores, get_text, metrics, file_path, description):
    """Process a checkpoint file whose data is a list (non-label version)."""
    print(f"Loading data from {file_path}")
    if file_path.endswith("pkl"):
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)
    elif file_path.endswith("json"):
        with open(file_path, 'r', encoding='utf-8') as file:
            loaded_data = json.load(file)
    elif file_path.endswith("jsonl"):
        infer_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                infer_data.append(json.loads(line.strip()))
        loaded_data = []
        for entry in infer_data:
            labels = entry.get('labels', [])
            loaded_data.append({
                "truth": entry['conversations'][1]['value'],
                "generated": entry['generated'],
                "labels": labels
            })
    elif file_path.endswith("csv"):
        import pandas as pd
        loaded_data = []
        df = pd.read_csv(file_path)
        truths = df['truth'].values
        generates = df['generated_11818_gradient123_mimic_chex'].values
        for truth, generate in zip(truths, generates):
            loaded_data.append({
                "truth": truth,
                "generated": generate
            })
    else:
        print("Unsupported file format.")
        return

    print(f"Loaded {len(loaded_data)} samples")
    # print("Sample:", loaded_data[0])

    # Extract epoch number using regex
    pattern = r"checkpoint-(\d+)\.pkl"
    match = re.search(pattern, file_path)
    if match:
        epoch_number = int(match.group(1))
        print(f"Epoch number: {epoch_number}")
    else:
        epoch_number = 1234567
        print("Epoch number not found in the file path.")

    refs = [each["truth"] for each in loaded_data]
    hyps = [get_text(each["generated"]) for each in loaded_data]
    compute_scores(
        metrics,
        refs,
        hyps,
        logger=logging.getLogger("test"),
        dump=True,
        epoch=epoch_number,
        description=description
    )


def process_checkpoint_label(compute_scores, get_text, metrics, file_path, description):
    """
    Process a checkpoint file whose data is a dictionary with keys corresponding
    to labels. For each label the scores are computed separately.
    """
    print(f"Loading data from {file_path}")
    if file_path.endswith("pkl"):
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)
    elif file_path.endswith("json"):
        with open(file_path, 'r', encoding='utf-8') as file:
            loaded_data = json.load(file)
    elif file_path.endswith("jsonl"):
        infer_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                infer_data.append(json.loads(line.strip()))
        loaded_data = {}
        # Assuming that each entry has a label key and you want to group by it.
        for entry in infer_data:
            label = entry.get("label", "default")
            if label not in loaded_data:
                loaded_data[label] = []
            loaded_data[label].append({
                "truth": entry['conversations'][1]['value'],
                "generated": entry['generated'],
                "labels": entry.get('labels', [])
            })
    elif file_path.endswith("csv"):
        import pandas as pd
        loaded_data = {}
        df = pd.read_csv(file_path)
        # Assuming the CSV has a column named 'label'
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            loaded_data[label] = []
            for truth, generate in zip(subset['truth'].values, subset['generated_11818_gradient123_mimic_chex'].values):
                loaded_data[label].append({
                    "truth": truth,
                    "generated": generate
                })
    else:
        print("Unsupported file format.")
        return

    # Print summary information
    print(f"Loaded {len(loaded_data)} keys")
    print("Keys:", list(loaded_data.keys()))

    # Extract epoch number. Updated regex to allow for "_label" in file name.
    pattern = r"checkpoint-(\d+)(?:_\w+)?\.pkl"
    match = re.search(pattern, file_path)
    if match:
        epoch_number = int(match.group(1))
        print(f"Epoch number: {epoch_number}")
    else:
        epoch_number = 1234567
        print("Epoch number not found in the file path.")

    # Process each label separately.
    for label_key, entries in loaded_data.items():
        print(f"Processing label: {label_key} with {len(entries)} samples")
        refs = [each["truth"] for each in entries]
        hyps = [get_text(each["generated"]) for each in entries]
        compute_scores(
            metrics,
            refs,
            hyps,
            logger=logging.getLogger("test"),
            dump=True,
            epoch=epoch_number,
            description=description,
            label=label_key
        )


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m vilmedic.blocks.scorers.combined_scores <description>")
        sys.exit(1)
    description = sys.argv[1]

    # Define the metrics you want to compute.
    metrics_normal = [
        "chexbert",
    ]

    metrics_label = [
        "BLEU",
        "METEOR",
        "CIDERD",
        "ROUGE1",
        "ROUGE2",
        "ROUGEL",
        "bertscore",
        # "chexbert",
        "radgraph",
        # "accuracy",
        # "f1-score",
        # "auroc",
        # "radentitymatchexact",
        # "radentitynli",
        # "stanford_ct_abd_accuracy",
    ]


    # Set your directory path here.
    directory_path = "/root/projects/InternVL-Epsi/internvl_chat/test_data/all_on_136_0202"

    # directory_path = "/root/projects/InternVL-Epsi/internvl_chat/test_data/26b_all_on_filter_0202"
    directory_path = "/root/projects/InternVL-Epsi/internvl_chat/test_data/26b_batch1_on_filter_0203"
    directory_path = "/root/projects/InternVL-Epsi/internvl_chat/test_data/26b_lunglesion_on_filter_0203"

    directory_path = "/root/projects/InternVL-Epsi/internvl_chat/test_data/26b_sixlabels_on_filter_0203"

    directory_path = "/root/projects/InternVL-Epsi/internvl_chat/test_data/26b_labels_on_random_laebl_0203"
    directory_path = "/root/projects/InternVL-Epsi/internvl_chat/test_data/26b_labels_on_random_laebl_0203-2/"
    directory_path = "/root/projects/InternVL-Epsi/internvl_chat/test_data/26b_labels_gradient_0722"

    directory_path = "/root/projects/InternVL-Epsi/internvl_chat/test_data/26b_continue_on_filter"

    # Get all .pkl files from the directory.
    pkl_files = glob.glob(os.path.join(directory_path, "*.pkl"))
    # Separate files with "label" in the filename from the others.
    normal_files = [fp for fp in pkl_files if "label" not in os.path.basename(fp)]
    label_files = [fp for fp in pkl_files if "label" in os.path.basename(fp)]

    # Sort the files by epoch number extracted from the filename.
    def sort_key_normal(fp):
        try:
            return int(re.search(r"checkpoint-(\d+)\.pkl", fp).group(1))
        except Exception:
            return 0

    def sort_key_label(fp):
        try:
            return int(re.search(r"checkpoint-(\d+)_label\.pkl", fp).group(1))
        except Exception:
            return 0

    normal_files = sorted(normal_files, key=sort_key_normal)
    label_files = sorted(label_files, key=sort_key_label)

    # Process non-label files.
    if normal_files:
        print("Processing non-label files:")
        for file_path in normal_files:
            print(f"Processing {file_path}")
            process_checkpoint_normal(compute_scores, get_text, metrics_normal, file_path, description)
    else:
        print("No non-label files found.")

    # Process label files.
    if label_files:
        print("Processing label files:")
        for file_path in label_files:
            print(f"Processing {file_path}")
            process_checkpoint_label(compute_scores, get_text, metrics_label, file_path, description)
    else:
        print("No label files found.")


if __name__ == "__main__":
    main()
