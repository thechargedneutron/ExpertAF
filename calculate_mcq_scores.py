'''
Assume scores are saved in scores/eval_results.pkl
'''

import pickle
import json
import sys

def ranking_metrics(all_labels_dict, all_scores_dict):
    import numpy as np

    # Assertion to check label correctness
    for labels in all_labels_dict.values():
        assert sum(labels) == 1 and len([l for l in labels if l == 1]) == 1, "Each sample must have exactly one label set to 1"

    recalls = {1: [], 3: [], 5: [], 10: [], 50: []}  # Adjust or add more ks as needed
    ranks = []

    # Calculate ranking metrics
    for sample, scores in all_scores_dict.items():
        labels = all_labels_dict[sample]
        assert sum(labels) == 1
        true_index = labels.index(1)

        # Sorting scores in descending order and getting the indices
        sorted_indices = np.argsort(scores)[::-1]

        # Rank of the correct label
        rank = np.where(sorted_indices == true_index)[0][0] + 1
        print(rank)
        ranks.append(rank)

        # Recall@k calculations
        for k in recalls:
            recalls[k].append(int(rank <= k))

    # Calculating average recall@k and median rank
    average_recalls = {k: np.mean(recalls[k]) * 100. for k in recalls}
    median_rank = np.median(ranks)

    return average_recalls, median_rank

egoexo_metadata_path = "/large_experiments/egoexo/v2/takes.json"
# Obtain various metadata
with open(egoexo_metadata_path) as f:
    egoexo_takes_metadata = json.load(f)
take2parenttask = {}
print("Processing metadata...")
for egoexo_take in egoexo_takes_metadata:
    take2parenttask[egoexo_take["take_name"]] = egoexo_take["parent_task_name"]

scores = []
for i in range(32):
    with open(f"scores/eval_results_True_False_False_{i}.pkl", "rb") as f:
        scores.extend(pickle.load(f))

with open(str(sys.argv[1])) as f:
    data = json.load(f)

index2score = {}
for i, datum in enumerate(scores):
    index2score[int(datum['inputs'])] = datum['loss']

# The index of the keys are the correct 'id' + positive or negative_idx so we can group by that
all_scores_dict = {}
all_labels_dict = {}
assert len(data) == 400*400 and len({item['id'] for item in data}) == 400*400
for i in range(len(data)):
    incorrect_take, incorrect_comm_idx, correct_take, correct_comm_idx, body_part, pos_or_neg = data[i]['id'].split('-----')
    if f'{incorrect_take}-----{incorrect_comm_idx}-----{correct_take}-----{correct_comm_idx}-----{body_part}' not in all_labels_dict:
        all_labels_dict[f'{incorrect_take}-----{incorrect_comm_idx}-----{correct_take}-----{correct_comm_idx}-----{body_part}'] = []
        all_scores_dict[f'{incorrect_take}-----{incorrect_comm_idx}-----{correct_take}-----{correct_comm_idx}-----{body_part}'] = []
    all_labels_dict[f'{incorrect_take}-----{incorrect_comm_idx}-----{correct_take}-----{correct_comm_idx}-----{body_part}'].append(1 if 'positive' in pos_or_neg else 0)
    all_scores_dict[f'{incorrect_take}-----{incorrect_comm_idx}-----{correct_take}-----{correct_comm_idx}-----{body_part}'].append(index2score[i])

average_recalls, median_rank = ranking_metrics(all_labels_dict, all_scores_dict)
print(average_recalls, median_rank)
exit()

acc = []
acc_bb = []
acc_s = []
acc_rc = []
num_options = 5
# increment the for loop in steps of num_options
for i in range(0, len(data), num_options):
    # Get all scores
    scores = []
    labels = []
    names = []
    for j in range(num_options):
        scores.append(index2score[i + j])
        labels.append(1 if data[i + j]['id'][-13:] == "-----positive" else 0)
        names.append(data[i + j]['id'][:-13] if data[i + j]['id'][-13:] == "-----positive" else data[i + j]['id'][:-15])
    # Assert all elements in names must be the same
    assert len(set(names)) == 1, "Incorrect order of data, unexpcted, please check"
    # Assert only one 1 in labels
    assert labels.count(1) == 1, "Incorrect order of data, unexpcted, please check"
    # Get the index of the max score and compare with the labels
    max_score_idx = scores.index(min(scores))
    acc.append(1 if labels[max_score_idx] == 1 else 0)
    if take2parenttask[names[0].split('-----')[0]] == "Basketball":
        acc_bb.append(1 if labels[max_score_idx] == 1 else 0)
    elif take2parenttask[names[0].split('-----')[0]] == "Soccer":
        acc_s.append(1 if labels[max_score_idx] == 1 else 0)
    elif take2parenttask[names[0].split('-----')[0]] == "Rock Climbing":
        acc_rc.append(1 if labels[max_score_idx] == 1 else 0)
    else:
        assert False, f"Unknown parent task: {take2parenttask[names[0].split('-----')[0]]}"

print("%^"*100)
print(f"Overall accuracy is: {sum(acc) / len(acc)}")
if len(acc_bb) > 0:
    print(f"Basketball accuracy is: {sum(acc_bb) / len(acc_bb)}")
if len(acc_s) > 0:
    print(f"Soccer accuracy is: {sum(acc_s) / len(acc_s)}")
if len(acc_rc) > 0:
    print(f"Rock Climbing accuracy is: {sum(acc_rc) / len(acc_rc)}")
