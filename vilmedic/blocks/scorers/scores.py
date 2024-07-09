import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from f1chexbert import F1CheXbert
from radgraph import F1RadGraph
from sklearn.metrics import classification_report, roc_auc_score

from . import *
from .utils import get_logger_directory

# RadGraph package overrides logger, need to set back to default
logging.setLoggerClass(logging.Logger)

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
):
    scores = dict()
    # If metric is None or empty list
    if metrics is None or not metrics:
        return scores

    assert (
        refs is not None and hyps is not None
    ), "You specified metrics but your evaluation does not return hyps nor refs"

    assert len(refs) == len(
        hyps
    ), "refs and hyps must have same length : {} vs {}".format(len(refs), len(hyps))

    # Dump
    if dump:
        print("---------------")
        print([logger])
        print([get_logger_directory(logger)])

        # base = os.path.join(
        #     get_logger_directory(logger), "{}_{}_{}".format(split, seed, "{}")
        # )
        base = f"logs/{split}_{seed}_{{}}"
        refs_file = base.format("refs.txt")
        hyps_file = base.format("hyps.txt")
        metrics_file = base.format("metrics.txt")

        with open(refs_file, "w") as f:
            f.write("\n".join(map(str, refs)))
            f.close()

        with open(hyps_file, "w") as f:
            f.write("\n".join(map(str, hyps)))
            f.close()

    for metric in metrics:
        print("Checking metric: ", metric)

        # metric_args = dict()
        #
        # # if metric has arguments
        # if OmegaConf.is_dict(metric):
        #     if len(metric) != 1:
        #         logger.warning("Metric badly formatted: {}. Skipping.".format(metric))
        #         continue
        #     metric_args = metric[list(metric.keys())[0]]
        #     metric = list(metric.keys())[0]

        # Iterating over metrics
        if metric == "BLEU":
            scores["BLEU"] = Bleu()(refs, hyps)[0]
        elif metric == "METEOR":
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.translate.meteor_score import meteor_score

            nltk.download("wordnet")
            nltk.download("punkt")

            tokenized_references = [word_tokenize(ref) for ref in refs]
            tokenized_hypotheses = [word_tokenize(hyp) for hyp in hyps]

            # Calculate METEOR score for each pair of reference and hypothesis
            metero_scores = [
                meteor_score([ref], hyp)
                for ref, hyp in zip(tokenized_references, tokenized_hypotheses)
            ]

            mean_score = np.mean(metero_scores)

            # scores["METEOR"] = Meteor()(refs, hyps)[0]
            scores["METEOR"] = mean_score
        elif metric == "CIDERD":
            scores["CIDERD"] = CiderD()(refs, hyps)[0]
        elif metric == "bertscore":
            scores["bertscore"] = BertScore()(refs, hyps)[0]
        elif metric in ["ROUGE1", "ROUGE2", "ROUGEL"]:
            scores[metric] = Rouge(rouges=[metric.lower()])(refs, hyps)[0]
        elif metric == "accuracy":
            scores["accuracy"] = round(
                np.mean(np.array(refs) == np.argmax(hyps, axis=-1)) * 100, 2
            )
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
            scores["chexbert-5_micro avg_f1-score"] = chexbert_5["micro avg"][
                "f1-score"
            ]
            scores["chexbert-all_micro avg_f1-score"] = chexbert_all["micro avg"][
                "f1-score"
            ]
            scores["chexbert-5_macro avg_f1-score"] = chexbert_5["macro avg"][
                "f1-score"
            ]
            scores["chexbert-all_macro avg_f1-score"] = chexbert_all["macro avg"][
                "f1-score"
            ]
        elif metric == "radentitymatchexact":
            scores["radentitymatchexact"] = RadEntityMatchExact()(refs, hyps)[0]
        elif metric == "radentitynli":
            scores["radentitynli"] = RadEntityNLI()(refs, hyps)[0]
        elif metric == "radgraph":
            (
                scores["radgraph_simple"],
                scores["radgraph_partial"],
                scores["radgraph_complete"],
            ) = F1RadGraph(reward_level="all", model_type="radgraph-xl")(
                refs=refs, hyps=hyps
            )[0]
        elif metric == "stanford_ct_abd_accuracy":
            scores["stanford_ct_abd"] = StanfordCTAbdAcc()(refs=refs, hyps=hyps)[0]
        else:
            logger.warning("Metric not implemented: {}".format(metric))

    if dump:
        with open(metrics_file, "a+") as f:
            f.write(
                json.dumps(
                    {"split": split, "epoch": epoch, "scores": scores},
                    indent=4,
                    sort_keys=False,
                )
            )
    return scores


def get_text(text):
    # Find the position of 'Assistant: '
    assistant_key = "Assistant: "
    start_idx = text.find(assistant_key)

    # Extract the text after 'Assistant: '
    if start_idx != -1:
        text = text[start_idx + len(assistant_key) :]

    return text


if __name__ == "__main__":
    import re

    metrics = [
        "BLEU",
        "METEOR",
        "CIDERD",
        "ROUGE1",
        "ROUGE2",
        "ROUGEL",
        "bertscore",
        # "accuracy",
        # "f1-score",
        # "auroc",
        "chexbert",
        # "radentitymatchexact",
        # "radentitynli",
        "radgraph",
        # "stanford_ct_abd_accuracy",
    ]

    import pickle

    file_path = "../CheXagent_k8s/infer_res_with_report_500_epoch_2.pkl"
    file_path = "../CheXagent_k8s/infer_res_with_report_1000_epoch_20.pkl"
    file_path = "../CheXagent_k8s/infer_res_with_report_500_epoch_10.pkl"

    file_path = "/mnt/data/ruian/idefics2/eval_res/infer_res_step_342_100.pkl"

    print(f"Loading data from {file_path}")

    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    # refs = [
    #     "The lung volumes are slightly low, with elevation of the right hemidiaphragm. There is mild peribronchial cuffing and engorgement of the pulmonary vasculature. Blunting of the costophrenic angles may represent small bilateral pleural effusions. There is no pneumothorax or consolidation concerning for pneumonia. The heart size is top-normal."
    # ]
    # # hyps = [
    # #     "The endotracheal tube has been removed. The cardiac and mediastinal silhouettes are stable. There is mild pulmonary vascular congestion. No focal consolidation, pleural effusion or pneumothorax is seen."
    # # ]

    # Regular expression pattern to extract the epoch number
    pattern = r"epoch_(\d+)\.pkl"

    # Search for the pattern in the file path
    match = re.search(pattern, file_path)

    # If a match is found, extract the epoch number
    if match:
        epoch_number = int(match.group(1))
        print(f"Epoch number: {epoch_number}")
    else:
        epoch_number = 1234567
        print("Epoch number not found in the file path.")

    # refs = [each['report_entry'] for each in loaded_data[:500]] # truth
    # hyps = [each['response'] for each in loaded_data[:500]] # prediction

    refs = [each["truth"] for each in loaded_data[:500]]  # truth
    hyps = [get_text(each["generated"]) for each in loaded_data[:500]]  # prediction

    compute_scores(
        metrics,
        refs,
        hyps,
        logger=logging.getLogger("test"),
        dump=True,
        epoch=epoch_number,
    )
