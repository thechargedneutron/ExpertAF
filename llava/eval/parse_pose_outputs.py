import json
import os
import numpy as np
import random
random.seed(0)

outputs = [json.loads(q) for q in open(os.path.expanduser("/path/to/LLaVA/output/answer-2.jsonl"), "r")]

from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, jaccard_score
import numpy as np

def evaluate_multilabel(y_true, y_pred):
    """
    Evaluates multiple multilabel classification metrics.
    
    Parameters:
    - y_true : array-like of shape (n_samples, n_classes)
      Ground truth (correct) label binary indicators.
    - y_pred : array-like of shape (n_samples, n_classes)
      Predicted labels, as returned by a classifier.
      
    Returns:
    A dictionary with metric names as keys and their scores as values.
    """
    results = {}
    
    # Calculating metrics
    results['Accuracy'] = accuracy_score(y_true, y_pred)
    results['Hamming Loss'] = hamming_loss(y_true, y_pred)
    results['Precision (macro)'] = precision_score(y_true, y_pred, average='macro')
    results['Recall (macro)'] = recall_score(y_true, y_pred, average='macro')
    results['F1 Score (macro)'] = f1_score(y_true, y_pred, average='macro')
    results['Jaccard Score (macro)'] = jaccard_score(y_true, y_pred, average='macro')
    
    # Micro average versions of precision, recall, and F1
    results['Precision (micro)'] = precision_score(y_true, y_pred, average='micro')
    results['Recall (micro)'] = recall_score(y_true, y_pred, average='micro')
    results['F1 Score (micro)'] = f1_score(y_true, y_pred, average='micro')
    results['Jaccard Score (micro)'] = jaccard_score(y_true, y_pred, average='micro')

    return results

gt = []
preds = []
random_baseline = False
for output in outputs:
    sample_gt = [0] * 5
    sample_pred = [0] * 5
    parts_to_idx = {'Head': 0, 'Hands': 1, 'Shoulder': 2, 'Legs': 3, 'Jump': 4}
    for part in parts_to_idx:
        if part in output['ground_truth']:
            sample_gt[parts_to_idx[part]] = 1
        if random_baseline:
            sample_pred[parts_to_idx[part]] = random.randint(0, 1)
        else:
            if part in output['text']:
                sample_pred[parts_to_idx[part]] = 1

    gt.append(sample_gt)
    preds.append(sample_pred)

gt = np.array(gt)
preds = np.array(preds)

results = evaluate_multilabel(gt, preds)
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")