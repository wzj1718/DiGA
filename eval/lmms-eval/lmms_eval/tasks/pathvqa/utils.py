import datetime
import json
import re
from collections import Counter

from loguru import logger as eval_logger
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


_ANSWER_PROCESSOR = EvalAIAnswerProcessor()
_YESNO = {"yes", "no"}


def pathvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def pathvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "\nAnswer with a short word or phrase.")
    return f"{pre_prompt}{doc['question'].strip()}{post_prompt}"


def _strip_answer_prefix(text):
    text = str(text).strip()
    text = text.split("\n", 1)[0].strip()
    text = re.sub(r"^(answer|the answer is|it is|it's)\s*[:\-]?\s*", "", text, flags=re.I).strip()
    return text


def _normalize_answer(text):
    text = _strip_answer_prefix(text)
    return _ANSWER_PROCESSOR(text)


def _yesno_answer(text):
    norm = _normalize_answer(text)
    if norm in _YESNO:
        return norm
    match = re.search(r"\b(yes|no)\b", norm)
    return match.group(1) if match else norm


def _token_f1(pred, target):
    pred_tokens = _normalize_answer(pred).split()
    target_tokens = _normalize_answer(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(target_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def _bleu(pred, target, n):
    pred_tokens = _normalize_answer(pred).split()
    target_tokens = _normalize_answer(target).split()
    if not pred_tokens or not target_tokens:
        return 0.0
    weights = tuple(1.0 / n for _ in range(n))
    return sentence_bleu([target_tokens], pred_tokens, weights=weights, smoothing_function=SmoothingFunction().method1)


def _make_record(doc, result):
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    raw_pred = result[0]
    target = doc["answer"]
    target_norm = _normalize_answer(target)
    is_yesno = target_norm in _YESNO
    pred_norm = _yesno_answer(raw_pred) if is_yesno else _normalize_answer(raw_pred)
    exact = float(pred_norm == target_norm)
    return {
        "question": doc["question"],
        "answer": target,
        "prediction": _strip_answer_prefix(raw_pred),
        "normalized_answer": target_norm,
        "normalized_prediction": pred_norm,
        "is_yesno": is_yesno,
        "exact": exact,
        "f1": _token_f1(raw_pred, target),
        "bleu1": _bleu(raw_pred, target, 1),
        "bleu2": _bleu(raw_pred, target, 2),
        "bleu3": _bleu(raw_pred, target, 3),
    }


def pathvqa_process_results(doc, result):
    record = _make_record(doc, result)
    return {
        "overall_acc": record,
        "yesno_acc": record,
        "freeform_acc": record,
        "macro_f1": record,
        "bleu1": record,
        "bleu2": record,
        "bleu3": record,
        "submission": record,
    }


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def pathvqa_aggregate_overall_acc(results, args=None):
    return _mean(item["exact"] for item in results)


def pathvqa_aggregate_yesno_acc(results, args=None):
    subset = [item["exact"] for item in results if item["is_yesno"]]
    return _mean(subset)


def pathvqa_aggregate_freeform_acc(results, args=None):
    subset = [item["exact"] for item in results if not item["is_yesno"]]
    return _mean(subset)


def pathvqa_aggregate_macro_f1(results, args=None):
    return _mean(item["f1"] for item in results)


def pathvqa_aggregate_bleu1(results, args=None):
    return _mean(item["bleu1"] for item in results)


def pathvqa_aggregate_bleu2(results, args=None):
    return _mean(item["bleu2"] for item in results)


def pathvqa_aggregate_bleu3(results, args=None):
    return _mean(item["bleu3"] for item in results)


def pathvqa_aggregate_submissions(results, args):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = generate_submission_file(f"pathvqa_predictions_{now}.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    eval_logger.info(f"Submission file saved to {path}")
    return 0.0
