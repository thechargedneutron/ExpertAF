import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
import transformers
from typing import Dict, Optional, Sequence, List

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_POSE1_TOKEN, DEFAULT_POSE2_TOKEN, POSE1_TOKEN_INDEX, POSE2_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, tokenizer_pose_token, process_images, get_model_name_from_path

from PIL import Image
import math

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# We do not expect any change in this class parameters, hence it is static.
class ModelArgs:
    def __init__(self):
        self.pose_weights = "/path/to/LLaVA/models/best_avg_joint_loss_epoch_150.pth"
        self.pose_module = "pct_tokenizer"
        self.pose_model_config = {
            "encoder": {
                "drop_rate": 0.2,
                "num_blocks": 4,
                "hidden_dim": 512,
                "token_inter_dim": 64,
                "hidden_inter_dim": 512,
                "dropout": 0.0
            },
            "decoder": {
                "num_blocks": 1,
                "hidden_dim": 32,
                "token_inter_dim": 64,
                "hidden_inter_dim": 64,
                "dropout": 0.0
            },
            "codebook": {
                "token_num": 34,
                "token_dim": 512,
                "token_class_num": 2048,
                "ema_decay": 0.9,
            }
        }
        self.pretrain_mm_mlp_adapter = ""
        self.mm_projector_type = "mlp2x_gelu"

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # Load the model_args
    tokenizer.model_max_length = 2048
    model_args = ModelArgs()
    model_args.pretrain_mm_mlp_adapter = args.pretrain_mm_mlp_adapter
    model_args.use_pose = args.use_pose
    model_args.use_egoexo = args.use_egoexo
    model_args.pose_generation = args.pose_generation
    model_args.tune_mm_mlp_adapter = False # doesn't matter
    if args.model_base is None:
        # If we are loading a full-FT model it already saves the correct tokenizer
        model.get_model().initialize_pose_modules(model_args=model_args)
        # Send to cuda
        model.get_model().get_pose_model().to('cuda').half()
        if args.use_pose:
            model.get_model().pose_mm_projector.to('cuda').half()
        if args.use_egoexo:
            model.get_model().ego_mm_projector.to('cuda').half()
            model.get_model().exo_mm_projector.to('cuda').half()

    if args.conv_mode == "llama_3" and args.model_base is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="<pad>"),
                    tokenizer=tokenizer,
                    model=model,
        )
    if args.model_base is None:
        model.initialize_pose_tokenizer(model_args, tokenizer)

    with open(args.question_file) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):

        # Needs improvement pose
        pose1_file = line['video1']
        start1 = eval(line['video1_range'])[0]
        ego1 = torch.load(os.path.join(args.egoexo_path, f"{pose1_file}_{start1}_aria.pth.tar"))
        exo1 = torch.load(os.path.join(args.egoexo_path, f"{pose1_file}_{start1}_exo.pth.tar"))
        if not args.pose_generation:
            pose1 = torch.tensor(np.load(os.path.join(args.pose_path, f"{pose1_file}.npy")))
        else:
            pose1 = torch.load(os.path.join(args.pose_token_path, f"{pose1_file}.pth.tar"))
        range1 = eval(line['video1_range'])
        pose1 = pose1[range1[0]:range1[1]]
        # Good execution pose
        pose2_file = line['video2']
        start2 = eval(line['video2_range'])[0]
        ego2 = torch.load(os.path.join(args.egoexo_path, f"{pose2_file}_{start2}_aria.pth.tar"))
        exo2 = torch.load(os.path.join(args.egoexo_path, f"{pose2_file}_{start2}_exo.pth.tar"))
        if not args.pose_generation:
            pose2 = torch.tensor(np.load(os.path.join(args.pose_path, f"{pose2_file}.npy")))
        else:
            pose2 = torch.load(os.path.join(args.pose_token_path, f"{pose2_file}.pth.tar"))
        range2 = eval(line['video2_range'])
        # dtw_best_frame_end_idx is incorrect, it uses whole 10s, don't use that
        pose2 = pose2[range2[0]:(range2[0] + range1[1] - range1[0])]
        # Sample pose frames uniformly from the two poses
        assert len(pose1) == len(pose2), "Pose lengths do not match"
        sample_idx = np.round(np.linspace(0, len(pose1) - 1, args.num_pose_frames)).astype(int)
        pose1 = pose1[sample_idx]
        pose2 = pose2[sample_idx]

        if args.pose_generation:
            # pose 1
            pose_string = ""
            assert pose1.ndim == 2, "Check input..."
            curr_ids = pose1
            for row_idx in range(len(curr_ids)):
                for col_idx in range(len(curr_ids[row_idx])):
                    assert int(curr_ids[row_idx][col_idx]) < 2048, "Why is the token index more than 2048?"
                    pose_string = pose_string + f"<CUSTOM_POSE_TOKEN_{int(curr_ids[row_idx][col_idx]):04}>"
                pose_string = pose_string + "<CUSTOM_POSE_TOKEN_FRAME_END>"
            pose_string = pose_string + "<CUSTOM_POSE_TOKEN_SEQ_END>"
            pose1_ids = pose_string
            assert pose_string.count('CUSTOM_POSE_TOKEN') == 351, "Hmmm....."
            # OLD definition
            # pose_string = ""
            # curr_ids = pose1
            # for idx in range(len(curr_ids)):
            #     assert int(curr_ids[idx]) < 2048, "Why is the token index more than 2048?"
            #     pose_string = pose_string + f"<CUSTOM_POSE_TOKEN_{int(curr_ids[idx]):04}>"
            # pose1_ids = pose_string

            # pose 2
            pose_string = ""
            assert pose2.ndim == 2, "Check input..."
            curr_ids = pose2
            for row_idx in range(len(curr_ids)):
                for col_idx in range(len(curr_ids[row_idx])):
                    assert int(curr_ids[row_idx][col_idx]) < 2048, "Why is the token index more than 2048?"
                    pose_string = pose_string + f"<CUSTOM_POSE_TOKEN_{int(curr_ids[row_idx][col_idx]):04}>"
                pose_string = pose_string + "<CUSTOM_POSE_TOKEN_FRAME_END>"
            pose_string = pose_string + "<CUSTOM_POSE_TOKEN_SEQ_END>"
            pose2_ids = pose_string
            assert pose_string.count('CUSTOM_POSE_TOKEN') == 351, "Hmmm....."
            # OLD definition
            # pose_string = ""
            # curr_ids = pose2.view(-1)
            # for idx in range(len(curr_ids)):
            #     assert int(curr_ids[idx]) < 2048, "Why is the token index more than 2048?"
            #     pose_string = pose_string + f"<CUSTOM_POSE_TOKEN_{int(curr_ids[idx]):04}>"
            # pose2_ids = pose_string

        idx = line["id"]
        if args.pose_generation:
            for line_idx in range(len(line['conversations'])):
                line['conversations'][line_idx]['value'] = line['conversations'][line_idx]['value'].replace('<pose1>', pose1_ids)
                line['conversations'][line_idx]['value'] = line['conversations'][line_idx]['value'].replace('<pose2>', pose2_ids)

        qs = line['conversations'][0]['value']

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if args.pose_generation:
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids
        else:
            input_ids = tokenizer_pose_token(prompt, tokenizer, POSE1_TOKEN_INDEX, POSE2_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            if type(model) == transformers.models.llama.modeling_llama.LlamaForCausalLM:
                output_ids = model.generate(
                input_ids.cuda(),
                do_sample=True if args.temperature > 0 else False,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True)
            else:
                output_ids = model.generate(
                    input_ids.cuda(),
                    images=None,#image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=None,#[image.size],
                    poses1=pose1.unsqueeze(0).half().cuda(),
                    egos1=ego1.unsqueeze(0).half().cuda(),
                    exos1=exo1.unsqueeze(0).half().cuda(),
                    poses2=pose2.unsqueeze(0).half().cuda(),
                    egos2=ego2.unsqueeze(0).half().cuda(),
                    exos2=exo2.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=(not args.pose_generation))[0].strip()
            del input_ids
            torch.cuda.empty_cache()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": outputs,
                                   "ground_truth": line['conversations'][1]['value'],
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--model-base", type=str, default=None) #/path/to/checkpoints-30-2-2e-3-cosine/checkpoint-549/
    parser.add_argument("--image-folder", type=str, default="/path/to/LLaVA/SEED-Bench-image/")
    parser.add_argument("--pose-path", type=str, default="/path/to/DatasetCurate/poses_finalized_v1/")
    parser.add_argument("--pose-token-path", type=str, default="/path/to/PCT_Tokenizer/tokens/")
    parser.add_argument("--egoexo-path", type=str, default="/path/to/InternVideo/InternVideo2/multi_modality/demo/egoexo4d_feats_with_soccer/")
    parser.add_argument("--question-file", type=str, default="/path/to/LLaVA/data/pose_dataset_concise_NE_DTW_with_soccer_val_N5.json")
    parser.add_argument("--pretrain-mm-mlp-adapter", type=str, default="/path/to/checkpoints-10-2-2e-2-cosine/checkpoint-798/mm_projector.bin")
    parser.add_argument("--answers-file", type=str, default="output/answer-llama3-mymethod.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--use-pose", type=boolean_string, default=False)
    parser.add_argument("--use-egoexo", type=boolean_string, default=True)
    parser.add_argument("--pose-generation", type=boolean_string, default=False)
    parser.add_argument("--num-pose-frames", type=int, default=10)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
