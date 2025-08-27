import os
from .clip_encoder import CLIPVisionTower
from .pct_tokenizer import PCT_Tokenizer


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_pose_model(pose_module, pose_model_config):
    if pose_module == "pct_tokenizer":
        return PCT_Tokenizer(
            "tokenizer",
            pose_model_config,
            num_joints=17,
            guide_ratio=0,
            guide_channels=0,
        )
    raise ValueError(f'Unknown pose module: {pose_module}')
