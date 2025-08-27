import json
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import os

import random
random.seed(0)

method = 'my'

def evaluate_text_scores(jsonl_file_path):
    with open(jsonl_file_path, 'r') as file:
        lines = file.readlines()
    
    # Prepare data for evaluation
    references = []
    candidates = []
    refs = {}
    cands = {}
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        reference = data['ground_truth']
        candidate = data['text']
        # assert len(reference.split("Here is the feedback based on the given pose sequence and the expert correct sequence: ")) == 2
        # assert len(candidate.split("Here is the feedback based on the given pose sequence and the expert correct sequence: ")) == 2
        # reference = reference.split("Here is the feedback based on the given pose sequence and the expert correct sequence: ")[1]
        # candidate = candidate.split("Here is the feedback based on the given pose sequence and the expert correct sequence: ")[1]
        
        # For BLEU and ROUGE
        references.append([reference.split()])  # BLEU expects tokenized reference as a list of lists
        candidates.append(candidate.split())     # Tokenized candidate
        
        # For CIDEr
        refs[str(i)] = [reference]
        cands[str(i)] = [candidate]
    
    # Compute BLEU score
    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(references, candidates, smoothing_function=smoothie)
    
    # Compute ROUGE score
    rouge = Rouge()
    rouge_scores = rouge.get_scores([" ".join(cand) for cand in candidates], [" ".join(ref[0]) for ref in references], avg=True)
    
    # Compute CIDEr score
    cider = Cider()
    (cider_score, _) = cider.compute_score(refs, cands)

    # Compute METEOR score
    meteor_scores = [meteor_score(ref, cand) for ref, cand in zip(references, candidates)]
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    
    # Calculate individual scores for ranking
    individual_bleus = [corpus_bleu([ref], [cand], smoothing_function=smoothie) for ref, cand in zip(references, candidates)]
    individual_rouges = [rouge.get_scores(" ".join(cand), " ".join(ref[0]))[0]['rouge-l']['f'] for ref, cand in zip(references, candidates)]
    _, individual_ciders = cider.compute_score(refs, cands)
    
    # Extract top 10 segments for each score
    top_bleu = sorted(zip(lines, individual_bleus), key=lambda x: x[1], reverse=True)[:10]
    top_rouge = sorted(zip(lines, individual_rouges), key=lambda x: x[1], reverse=True)[:10]
    top_cider = sorted(zip(lines, individual_ciders), key=lambda x: x[1], reverse=True)[:10]

    return {
        'bleu_score': bleu_score,
        'rouge_scores': rouge_scores,
        'cider_score': cider_score,
        'meteor_score': avg_meteor_score,
        # 'top_bleu': top_bleu,
        # 'top_rouge': top_rouge,
        # 'top_cider': top_cider
    }

import sys
if method != 'random':
    result = evaluate_text_scores(sys.argv[1])
    print(result)






def evaluate_random_text_scores(answers, ground_truths):
    
    # Prepare data for evaluation
    references = []
    candidates = []
    refs = {}
    cands = {}
    
    for i in range(len(answers)):
        reference = ground_truths[i]
        candidate = answers[i]

        # assert len(reference.split("Here is the feedback based on the given pose sequence and the expert correct sequence: ")) == 2
        # assert len(candidate.split("Here is the feedback based on the given pose sequence and the expert correct sequence: ")) == 2
        # reference = reference.split("Here is the feedback based on the given pose sequence and the expert correct sequence: ")[1]
        # candidate = candidate.split("Here is the feedback based on the given pose sequence and the expert correct sequence: ")[1]
        
        # For BLEU and ROUGE
        references.append([reference.split()])  # BLEU expects tokenized reference as a list of lists
        candidates.append(candidate.split())     # Tokenized candidate
        
        # For CIDEr
        refs[str(i)] = [reference]
        cands[str(i)] = [candidate]
    
    # Compute BLEU score
    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(references, candidates, smoothing_function=smoothie)
    
    # Compute ROUGE score
    rouge = Rouge()
    rouge_scores = rouge.get_scores([" ".join(cand) for cand in candidates], [" ".join(ref[0]) for ref in references], avg=True)
    
    # Compute CIDEr score
    cider = Cider()
    (cider_score, _) = cider.compute_score(refs, cands)

    # Compute METEOR score
    meteor_scores = [meteor_score(ref, cand) for ref, cand in zip(references, candidates)]
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    
    # Calculate individual scores for ranking
    individual_bleus = [corpus_bleu([ref], [cand], smoothing_function=smoothie) for ref, cand in zip(references, candidates)]
    individual_rouges = [rouge.get_scores(" ".join(cand), " ".join(ref[0]))[0]['rouge-l']['f'] for ref, cand in zip(references, candidates)]
    _, individual_ciders = cider.compute_score(refs, cands)
    
    # Extract top 10 segments for each score
    # top_bleu = sorted(zip(lines, individual_bleus), key=lambda x: x[1], reverse=True)[:10]
    # top_rouge = sorted(zip(lines, individual_rouges), key=lambda x: x[1], reverse=True)[:10]
    # top_cider = sorted(zip(lines, individual_ciders), key=lambda x: x[1], reverse=True)[:10]

    return {
        'bleu_score': bleu_score,
        'rouge_scores': rouge_scores,
        'cider_score': cider_score,
        'meteor_score': avg_meteor_score,
        # 'top_bleu': top_bleu,
        # 'top_rouge': top_rouge,
        # 'top_cider': top_cider
    }

def random_retrieval_metrics():
    # Get test sentences
    baseline_sents_posescript = []
    ground_truth_sents = []
    with open("/path/to/LLaVA/data/pose_dataset_concise_NE_DTW_with_soccer_val_N5.json") as f:
        questions = json.load(f)
    
    for line in questions:
        # print(line)
        posescript_path = '/path/to/posescript/src/text2pose/egoexo_captions/'
        potential_file = f"{line['video1']}_{eval(line['video1_range'])[0]}"
        potential_captions = [x for x in os.listdir(posescript_path) if potential_file in x]
        if len(potential_captions) == 0: print('[WARN]') # only one file missing / 1272
        if len(potential_captions) > 0:
            chosen_path = random.sample(potential_captions, 1)[0]
            with open(os.path.join(posescript_path, chosen_path)) as fr:
                sentence = fr.read()
        else:
            sentence = ""
        sentence = "Here is the feedback based on the given pose sequence and the expert correct sequence: " + sentence
        ground_truth_sents.append(line['conversations'][1]['value'])
        baseline_sents_posescript.append(sentence)

    # Get random train sentences
    baseline_sents = []
    with open("/path/to/LLaVA/data/pose_dataset_concise_NE_DTW_with_soccer_val_N5.json") as f:
        questions = json.load(f)
    for i in range(len(ground_truth_sents)):
        line = random.choice(questions)
        baseline_sents.append(line['conversations'][1]['value'])
    
    results = evaluate_random_text_scores(baseline_sents_posescript, ground_truth_sents)
    print(results)

if method == 'random':
    random_retrieval_metrics()