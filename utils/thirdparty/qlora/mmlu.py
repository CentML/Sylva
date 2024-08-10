# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Source: https://github.com/artidoro/qlora/tree/main

import evaluate
import numpy as np
import torch


def mmlu_eval(model, tokenizer, valid_dataloader, device, verbose=False):
    IGNORE_INDEX = -100

    accuracy = evaluate.load("accuracy")
    preds, refs = [], []
    loss_mmlu = 0
    label_names = ["labels"]
    ignore_keys = ["past_key_values"]
    abcd_idx = [
        tokenizer("A", add_special_tokens=False).input_ids[0],
        tokenizer("B", add_special_tokens=False).input_ids[0],
        tokenizer("C", add_special_tokens=False).input_ids[0],
        tokenizer("D", add_special_tokens=False).input_ids[0],
    ]

    for batch in valid_dataloader:
        with torch.no_grad():
            batch = to_device(batch, device)
            labels = nested_detach(tuple(batch.get(name) for name in label_names))
            if len(labels) == 1:
                labels = labels[0]
            with torch.autocast("cuda"):
                outputs = model(**batch)
            loss = outputs.loss
            loss = loss.mean().detach()
            logits = tuple(
                v for k, v in outputs.items() if k not in ignore_keys + ["loss"]
            )
            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]

        # two tokens: output and eos token
        for i, logit in enumerate(logits):
            label_non_zero_id = (batch["labels"][i] != IGNORE_INDEX).nonzero()[0][0]
            logit_abcd = logit[label_non_zero_id - 1][abcd_idx]
            preds.append(torch.argmax(logit_abcd).item())
        labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:, 0]
        refs += [abcd_idx.index(label) for label in labels.tolist()]
        loss_mmlu += loss.item()

    results = {"mmlu_loss": loss_mmlu / len(valid_dataloader)}
    subject = valid_dataloader.dataset["subject"]
    subjects = {s: {"refs": [], "preds": []} for s in set(subject)}
    for s, p, r in zip(subject, preds, refs):
        subjects[s]["preds"].append(p)
        subjects[s]["refs"].append(r)
    subject_scores = []
    for subject in subjects:
        subject_score = accuracy.compute(
            references=subjects[subject]["refs"], predictions=subjects[subject]["preds"]
        )["accuracy"]
        results[f"mmlu_eval_accuracy_{subject}"] = subject_score
        subject_scores.append(subject_score)
    results["mmlu_eval_accuracy"] = np.mean(subject_scores)

    if verbose:
        print("****** MMLU EVAL ******")
        for k, v in results.items():
            if "accuracy" in k:
                print(f"{k}: {v * 100:.2f}")
            else:
                print(f"{k}: {v:.2f}")

    return results["mmlu_eval_accuracy"], results["mmlu_loss"], results


def to_device(obj, device=None):
    if hasattr(obj, "items"):
        output = {}
        for k, v in obj.items():
            if device:
                output[k] = v.to(device)
            else:
                output[k] = v.cuda()
    else:
        if device:
            output = obj.to(device)
        else:
            output = obj.cuda()
    return output


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    from collections.abc import Mapping

    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()
