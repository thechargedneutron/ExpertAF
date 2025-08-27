import json
from tqdm import tqdm
import numpy as np
import re
import torch
import matplotlib.pyplot as plt

from llava.model.multimodal_encoder.pct_tokenizer import PCT_Tokenizer

def create_skeleton_coco_format():
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle')
    }
    left_group_color = [2, 62, 138]
    left_hand_color = [106, 143, 95]
    face_color = [128, 0, 0]
    right_group_color = [121, 48, 90]
    right_hand_color = [129, 128, 0]
    skeleton_info_new = {
        0: dict(link=('left_shoulder', 'left_elbow'), id=0, color=left_hand_color),
        1: dict(link=('left_elbow', 'left_wrist'), id=1, color=left_hand_color),
        2: dict(link=('right_shoulder', 'right_elbow'), id=2, color=right_hand_color),
        3: dict(link=('right_elbow', 'right_wrist'), id=3, color=right_hand_color),
        4: dict(link=('left_shoulder', 'right_shoulder'), id=4, color=left_hand_color),
        5: dict(link=('right_shoulder', 'right_hip'), id=5, color=right_group_color),
        6: dict(link=('right_hip', 'right_knee'), id=6, color=right_group_color),
        7: dict(link=('right_knee', 'right_ankle'), id=7, color=right_group_color),
        8: dict(link=('left_shoulder', 'left_hip'), id=8, color=left_group_color),
        9: dict(link=('left_hip', 'left_knee'), id=9, color=left_group_color),
        10: dict(link=('left_knee', 'left_ankle'), id=10, color=left_group_color),
        11: dict(link=('left_eye', 'right_eye'), id=11, color=face_color),
        12: dict(link=('left_eye', 'left_ear'), id=12, color=face_color),
        13: dict(link=('right_eye', 'right_ear'), id=13, color=face_color),
        14: dict(link=('nose', 'left_eye'), id=14, color=face_color),
        15: dict(link=('nose', 'right_eye'), id=15, color=face_color)
    }
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255])
    }

    # Create point color list
    points_colors = []
    joint_locations = []
    for key in keypoint_info:
        points_colors.append(np.array(keypoint_info[key]['color']) / 255.)
        joint_locations.append(keypoint_info[key]['name'])

    skeleton = []
    for key in skeleton_info_new:
        skeleton.append((joint_locations.index(skeleton_info_new[key]['link'][0]), joint_locations.index(skeleton_info_new[key]['link'][1]), skeleton_info_new[key]['link'][0], skeleton_info_new[key]['link'][1], np.array(skeleton_info_new[key]['color']) / 255.))

    return skeleton, points_colors

def plot_pose_image(poses, frame_index=-1, filename='pose_frame.png', return_string=False):
    if poses.ndim == 2:
        poses = np.expand_dims(poses, axis=0)
        frame_index = 0
    if poses.ndim == 3:
        assert frame_index != -1
    else:
        raise ValueError
    if poses.shape[-1] > 3:
        poses = poses[:, :, :3]
    # Get COCO format data
    skeleton, points_colors = create_skeleton_coco_format()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each bone in the skeleton
    for i, j, joint_i, joint_j, curr_color in skeleton:
        ax.plot([poses[frame_index, i, 0], poses[frame_index, j, 0]],
                [poses[frame_index, i, 1], poses[frame_index, j, 1]],
                [poses[frame_index, i, 2], poses[frame_index, j, 2]], color=curr_color, marker='o', markersize=1)
        # print(i, j, joint_i, joint_j, curr_color)
    # exit()

    # Automatically adjust the axes limits
    x_span = poses[:, :, 0].max() - poses[:, :, 0].min()
    y_span = poses[:, :, 1].max() - poses[:, :, 1].min()
    xy_span = max(x_span, y_span)
    ax.auto_scale_xyz([poses[:, :, 0].min(), poses[:, :, 0].min() + xy_span],
                      [poses[:, :, 1].min(), poses[:, :, 1].min() + xy_span],
                      [poses[:, :, 2].min(), poses[:, :, 2].max()])

    # Turn off the grid
    ax.grid(False)

    # Hide the axes lines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Hide x axis line
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Hide y axis line
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Hide z axis line

    # Hide the panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Hide the axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Hide the axes labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Hide the axes spines by setting their visibility to False
    ax.w_xaxis.pane.set_visible(False)
    ax.w_yaxis.pane.set_visible(False)
    ax.w_zaxis.pane.set_visible(False)

    if return_string:
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        fig.savefig(pngImage, format="png")
        # Encode PNG image to base64 string
        pngImageB64String = base64.b64encode(pngImage.getvalue()).decode('utf8')
        return pngImageB64String
    else:
        # Save the figure
        plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free memory

        print(f"Image saved as {filename}")

# Define the tokenizer
pose_model_config = {
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

tokenizer = PCT_Tokenizer(
            "tokenizer",
            pose_model_config,
            num_joints=17,
            guide_ratio=0,
            guide_channels=0,
        )

pose_weights = "/path/to/LLaVA/models/best_avg_joint_loss_epoch_150.pth"
pose_weights = torch.load(pose_weights, map_location='cpu')
prefix = 'keypoint_head.tokenizer.'
modified_state_dict = {key[len(prefix):] if key.startswith(prefix) else key: value
                for key, value in pose_weights['state_dict'].items()}
tokenizer.load_state_dict(modified_state_dict, strict=True)
print("Successfully loaded the model with weights...")
###########################

# Loss function definition
def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    return re

def parse_tokens(token_string):
    # Regular expressions to match custom pose tokens, frame end, and sequence end tokens
    token_pattern = re.compile(r'<CUSTOM_POSE_TOKEN_(\d{4})>')
    frame_end_pattern = re.compile(r'<CUSTOM_POSE_TOKEN_FRAME_END>')
    seq_end_pattern = re.compile(r'<CUSTOM_POSE_TOKEN_SEQ_END>')

    # Initialize variables
    frames = []
    current_frame = []

    # Iterate over all matches in the input string
    for match in re.finditer(r'<CUSTOM_POSE_TOKEN_(\d{4})>|<CUSTOM_POSE_TOKEN_FRAME_END>|<CUSTOM_POSE_TOKEN_SEQ_END>', token_string):
        token = match.group(0)

        # Process custom pose token
        token_match = token_pattern.match(token)
        if token_match:
            token_number = int(token_match.group(1))
            current_frame.append(token_number)
            continue

        # Process frame end token
        if frame_end_pattern.match(token):
            if current_frame:
                frames.append(current_frame)
                current_frame = []
            continue

        # Process sequence end token
        if seq_end_pattern.match(token):
            if current_frame:
                frames.append(current_frame)
            break

    return frames

def find_min_k_indices(arr, k):
    arr_cleaned = np.where(np.isnan(arr), np.inf, arr)
    partitioned_indices = np.argpartition(arr_cleaned, k)[:k]
    sorted_indices = partitioned_indices[np.argsort(arr_cleaned[partitioned_indices])]
    return sorted_indices


basefile = 'output/pose_gen_pose_mymethod_3816_{}_of_16.jsonl'
recon_errors = []
ground_truth_joints_collection = []
prediction_joints_collection = []
input_joints_collection = []
commentaries_collection = []
total_generations = 0
for idx in tqdm(range(16)):
    with open(basefile.format(idx)) as f:
        result = [json.loads(jline) for jline in f.readlines()]
    total_generations += len(result)
    for sample_idx in range(len(result)):
        ground_truth = parse_tokens(result[sample_idx]['ground_truth'])
        ground_truth = torch.tensor(ground_truth)
        bsz = ground_truth.shape[0]
        ground_truth_joints = tokenizer.decode_tokens(ground_truth.view(-1), bsz)
        assert len(result[sample_idx]['text'].split('<|start_header_id|>assistant<|end_header_id|>')) == 2, f"Why are there two assistant responses?, len is {0}"
        commentary = result[sample_idx]['text'].split('<|start_header_id|>assistant<|end_header_id|>')[0].split("Here is a pose sequence of a person doing")[-1].split("Tell me the correct way of doing this step based on an expert's feedback given as follows:")[-1].split('<|eot_id|>')[0]
        commentaries_collection.append(commentary)
        input_pose = parse_tokens(result[sample_idx]['text'].split('<|start_header_id|>assistant<|end_header_id|>')[0])
        input_pose = torch.tensor(input_pose)
        prediction = parse_tokens(result[sample_idx]['text'].split('<|start_header_id|>assistant<|end_header_id|>')[-1])
        prediction = torch.tensor(prediction)
        bsz = prediction.shape[0]
        prediction_joints = tokenizer.decode_tokens(prediction.view(-1), bsz)
        bsz = input_pose.shape[0]
        input_joints = tokenizer.decode_tokens(input_pose.view(-1), bsz)
        recon_error = reconstruction_error(prediction_joints, ground_truth_joints)
        recon_errors.append(np.mean(recon_error.detach().numpy()))
        ground_truth_joints_collection.append(ground_truth_joints)
        prediction_joints_collection.append(prediction_joints)
        input_joints_collection.append(input_joints)

recon_errors = np.array(recon_errors)
print(recon_error.shape)
print(recon_error)
print(recon_errors.shape)
error = np.nanmean(recon_errors)
print(error)
print(np.min(recon_errors))
exit()

# visualize best k samples
import random
random.seed(0)

k = 500
best_indices = find_min_k_indices(recon_errors, k)
viz_samples = random.sample(list(best_indices), 20)
for save_idx, sampled_idx in enumerate(viz_samples):
    print(f"Saving {save_idx}/{20}...")
    print(commentaries_collection[sampled_idx])
    print("\n")
    gt = ground_truth_joints_collection[sampled_idx].detach().numpy()
    pred = prediction_joints_collection[sampled_idx].detach().numpy()
    ip = input_joints_collection[sampled_idx].detach().numpy()
    frame_idx = random.randint(0, len(gt) - 1)
    plot_pose_image(gt, frame_idx, f'pose_reconstruction_viz/gt_{save_idx}.png')
    plot_pose_image(pred, frame_idx, f'pose_reconstruction_viz/pred_{save_idx}.png')
    plot_pose_image(ip, frame_idx, f'pose_reconstruction_viz/input_{save_idx}.png')
exit()
        # print(prediction.shape)