"""Self-contained re-implementation of the SEAR model.

Ported 1:1 from the authors' reference implementation
(https://github.com/Schindler-EPFL-Lab/SEAR):

    sear/models/thermal_aggregators/base.py           -> ThermalAggregatorBase
    sear/models/thermal_aggregators/lora.py           -> ThermalAggregatorLoRA
    sear/models/thermal_aggregators/thermal_camera_token.py
                                                      -> ThermalAggregatorThermalCameraToken
    sear/models/thermal_aggregators/custom_patterns.py -> CustomPatterns
    sear/models/thermal_vggt.py                       -> ThermalVGGT

The only deviation is that the PyTorch-Lightning wrapper
(`sear/models/vggt_wrapper.py`) is replaced by the plain-torch `load_sear_model`
helper below: Lightning is only needed for training / metrics, and its
`on_load_checkpoint` hook does nothing more than forward the two saved tensors
groups (`lora`, `thermal_camera_token`) into `ThermalVGGT.load_state_dict`.
"""

from __future__ import annotations

import abc
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten
from vggt.models.vggt import CameraHead, DPTHead


class CustomPatterns(Enum):
    """Which modules of the Alternating-Attention stack receive LoRA."""

    FRAME_ONLY = r"^frame_blocks.+"
    GLOBAL_ONLY = r"^global_blocks.+"
    ALL_BLOCKS = r"(^frame_blocks.+)|(^global_blocks.+)"
    FIRST_QUATER = r"^(?:frame|global)_blocks\.[0-5](\..*|$)"
    FIRST_TWO_QUATERS = r"^(?:frame|global)_blocks\.([0-9]|1[0-1])(\..*|$)"
    FIRST_THREE_QUATERS = r"^(?:frame|global)_blocks\.([0-9]|1[0-7])(\..*|$)"


class ThermalAggregatorBase(nn.Module, abc.ABC):
    """Wrapper around the VGGT aggregator that also handles thermal frames."""

    KEEP_IN_STATE_DICT: set[str] = set()

    def __init__(
        self,
        vggt_state_dict: Mapping[str, Any],
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )
        self.aggregator.load_state_dict(
            state_dict=self.get_state_dict_part(
                state_dict=vggt_state_dict, starts_with="aggregator"
            )
        )
        self._embed_dim = embed_dim

    @abc.abstractmethod
    def process_thermal_tokens(
        self,
        B: int,
        S: int,
        tokens: torch.Tensor,
        thermal_mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get_state_dict_part(
        state_dict: Mapping[str, Any], starts_with: str
    ) -> Mapping[str, Any]:
        """Sub-state-dict of `state_dict` for keys under `starts_with`, prefix stripped."""
        n = len(starts_with)
        return {
            k[n + 1 :]: v for k, v in state_dict.items() if k.startswith(starts_with)
        }

    @staticmethod
    def _find_injectable_layers(
        pattern: "re.Pattern[str]",
        module: nn.Module,
        module_name: str = "",
        root: bool = True,
    ) -> list[str]:
        injectable_layer_classes = (
            nn.Linear,
            nn.Embedding,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
        )
        if pattern.match(module_name) and isinstance(module, injectable_layer_classes):
            return [module_name]

        result: list[str] = []
        for name, layer in module.named_children():
            child_name = name if root else f"{module_name}.{name}"
            result.extend(
                ThermalAggregatorBase._find_injectable_layers(
                    pattern=pattern, module=layer, module_name=child_name, root=False
                )
            )
        return result

    def forward(
        self, images: torch.Tensor, thermal_mask: torch.Tensor
    ) -> tuple[list[torch.Tensor], int]:
        """VGGT aggregator forward, with thermal frames handled separately.

        `images` is [B, S, 3, H, W], `thermal_mask` is [B, S] of booleans.
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images = (images - self.aggregator._resnet_mean) / self.aggregator._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.aggregator.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        camera_token = slice_expand_and_flatten(self.aggregator.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.aggregator.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        # --- the only difference with the original VGGT aggregator forward ---
        thermal_mask_flat = thermal_mask.view(B * S)
        if torch.any(thermal_mask_flat):
            tokens = self.process_thermal_tokens(
                B=B, S=S, tokens=tokens, thermal_mask_flat=thermal_mask_flat
            )
        # ---------------------------------------------------------------------

        position = None
        if self.aggregator.position_getter is not None:
            position = self.aggregator.position_getter(
                B * S,
                H // self.aggregator.patch_size,
                W // self.aggregator.patch_size,
                device=images.device,
            )

        if self.aggregator.patch_start_idx > 0 and position is not None:
            position = position + 1
            pos_special = (
                torch.zeros(B * S, self.aggregator.patch_start_idx, 2)
                .to(images.device)
                .to(position.dtype)
            )
            position = torch.cat([pos_special, position], dim=1)

        _, num_of_patches, embedding_dimension = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list: list[torch.Tensor] = []
        frame_intermediates = None
        global_intermediates = None

        for _ in range(self.aggregator.aa_block_num):
            for attn_type in self.aggregator.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = (
                        self.aggregator._process_frame_attention(
                            tokens,
                            B,
                            S,
                            num_of_patches,
                            embedding_dimension,
                            frame_idx,
                            pos=position,
                        )
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = (
                        self.aggregator._process_global_attention(
                            tokens,
                            B,
                            S,
                            num_of_patches,
                            embedding_dimension,
                            global_idx,
                            pos=position,
                        )
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            assert frame_intermediates is not None
            assert global_intermediates is not None
            for i in range(len(frame_intermediates)):
                output_list.append(
                    torch.cat(
                        [frame_intermediates[i], global_intermediates[i]], dim=-1
                    )
                )

        del frame_intermediates
        del global_intermediates
        return output_list, self.aggregator.patch_start_idx


class ThermalAggregatorLoRA(ThermalAggregatorBase):
    """Aggregator with LoRA injected into the blocks selected by `pattern`."""

    KEEP_IN_STATE_DICT: set[str] = {"lora"}

    def __init__(
        self,
        pattern: CustomPatterns,
        vggt_state_dict: Mapping[str, Any],
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        lora_alpha: int = 128,
        lora_rank: int = 64,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            vggt_state_dict=vggt_state_dict,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        re_pattern = re.compile(pattern.value)
        target_modules = self._find_injectable_layers(
            pattern=re_pattern, module=self.aggregator
        )

        self._lora_alpha = lora_alpha
        self._lora_rank = lora_rank
        self._lora_dropout = lora_dropout

        lora_config = LoraConfig(
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            r=self._lora_rank,
            bias="none",
            target_modules=target_modules,
        )
        self.aggregator = inject_adapter_in_model(
            peft_config=lora_config, model=self.aggregator
        )

    def process_thermal_tokens(
        self,
        B: int,
        S: int,
        tokens: torch.Tensor,
        thermal_mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        return tokens

    def state_dict(self, *args, destination=None, **kwargs):  # type: ignore[override]
        if destination is None:
            destination = {}
        destination["lora"] = get_peft_model_state_dict(self.aggregator)
        return destination

    def load_state_dict(self, state_dict, *args, **kwargs) -> None:  # type: ignore[override]
        set_peft_model_state_dict(self.aggregator, state_dict["lora"])


class ThermalAggregatorThermalCameraToken(ThermalAggregatorLoRA):
    """SEAR: LoRA + a learnable <thermal camera token> for thermal frames."""

    KEEP_IN_STATE_DICT: set[str] = {"thermal_camera_token", "lora"}

    def __init__(
        self,
        vggt_state_dict: Mapping[str, Any],
        pattern: CustomPatterns = CustomPatterns.ALL_BLOCKS,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        lora_alpha: int = 128,
        lora_rank: int = 64,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            vggt_state_dict=vggt_state_dict,
            pattern=pattern,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            lora_alpha=lora_alpha,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
        )
        self._thermal_camera_token = nn.Parameter(
            self.aggregator.camera_token.detach().clone(), requires_grad=True
        )

    def process_thermal_tokens(
        self,
        B: int,
        S: int,
        tokens: torch.Tensor,
        thermal_mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        thermal_camera_token = slice_expand_and_flatten(
            token_tensor=self._thermal_camera_token, B=B, S=S
        )
        tokens = tokens.clone()
        tokens[thermal_mask_flat, 0, :] = thermal_camera_token[thermal_mask_flat, 0, :]
        return tokens

    def state_dict(self, *args, destination=None, **kwargs):  # type: ignore[override]
        destination = ThermalAggregatorLoRA.state_dict(self, destination=destination)
        destination["thermal_camera_token"] = self._thermal_camera_token.detach()
        return destination

    def load_state_dict(self, state_dict, *args, **kwargs) -> None:  # type: ignore[override]
        ThermalAggregatorLoRA.load_state_dict(self, state_dict=state_dict)
        self._thermal_camera_token.data.copy_(state_dict["thermal_camera_token"])


class ThermalVGGT(nn.Module):
    """VGGT with the SEAR thermal aggregator; camera + depth heads only."""

    def __init__(
        self,
        vggt_state_dict: Mapping[str, Any],
        thermal_aggregator: ThermalAggregatorBase,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.aggregator = thermal_aggregator

        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
        )

        self.camera_head.load_state_dict(
            state_dict=ThermalAggregatorBase.get_state_dict_part(
                state_dict=vggt_state_dict, starts_with="camera_head"
            )
        )
        self.depth_head.load_state_dict(
            state_dict=ThermalAggregatorBase.get_state_dict_part(
                state_dict=vggt_state_dict, starts_with="depth_head"
            )
        )

        for p in self.camera_head.parameters():
            p.requires_grad_(False)
        for p in self.depth_head.parameters():
            p.requires_grad_(False)

    def forward(
        self, images: torch.Tensor, thermal_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """`images` is [B, S, 3, H, W]; `thermal_mask` is [B, S] booleans."""
        if images.ndim != 5:
            raise RuntimeError(
                f"`images` must be of size [B, S, 3, H, W] but has ndim = {images.ndim}"
            )
        if thermal_mask.ndim != 2:
            raise RuntimeError(
                f"`thermal_mask` must be of size [B, S] but has ndim = {thermal_mask.ndim}"
            )
        if images.shape[:2] != thermal_mask.shape[:2]:
            raise RuntimeError(
                "`images` and `thermal_mask` have mismatched shapes: "
                f"images {images.shape}, thermal {thermal_mask.shape}"
            )

        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images=images, thermal_mask=thermal_mask
        )

        predictions: dict[str, Any] = {}
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list

            depth, depth_conf = self.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        predictions["images"] = images
        return predictions

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.aggregator.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs) -> None:  # type: ignore[override]
        self.aggregator.load_state_dict(*args, **kwargs)

def load_sear_model(
    sear_repo_id: str = "MalcolmMielle/SEAR",
    sear_filename: str = "SEAR.ckpt",
    vggt_repo_id: str = "facebook/VGGT-1B",
    vggt_filename: str = "model.pt",
) -> ThermalVGGT:
    """Builds SEAR: the VGGT-1B backbone plus the released LoRA + thermal token."""
    from huggingface_hub import hf_hub_download

    vggt_path = hf_hub_download(repo_id=vggt_repo_id, filename=vggt_filename)
    vggt_state_dict = torch.load(vggt_path, map_location="cpu", weights_only=True)

    aggregator = ThermalAggregatorThermalCameraToken(
        vggt_state_dict=vggt_state_dict,
        pattern=CustomPatterns.ALL_BLOCKS,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        lora_alpha=128,
        lora_rank=64,
        lora_dropout=0.1,
    )
    model = ThermalVGGT(
        vggt_state_dict=vggt_state_dict, thermal_aggregator=aggregator, embed_dim=1024
    )
    del vggt_state_dict

    sear_path = hf_hub_download(repo_id=sear_repo_id, filename=sear_filename)
    checkpoint = torch.load(sear_path, map_location="cpu", weights_only=True)
    state = {
        k: v
        for k, v in checkpoint["state_dict"].items()
        if k in aggregator.KEEP_IN_STATE_DICT
    }
    missing = aggregator.KEEP_IN_STATE_DICT - set(state)
    if missing:
        raise RuntimeError(f"SEAR checkpoint is missing entries: {sorted(missing)}")
    model.load_state_dict(state)
    del checkpoint, state

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

## Stuff to run SEAR
import time  # noqa: E402
import numpy as np

from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
MAX_FRAMES = 512

def _subsample(paths: list[str], budget: int) -> list[str]:
    if len(paths) <= budget or budget <= 0:
        return paths
    idx = np.linspace(0, len(paths) - 1, budget).round().astype(int)
    return [paths[i] for i in sorted(set(idx.tolist()))]

def _paths(files) -> list[str]:
    """Normalise whatever gr.File hands us into a sorted list of paths."""
    if files is None:
        return []
    if not isinstance(files, (list, tuple)):
        files = [files]
    out = []
    for f in files:
        if f is None:
            continue
        if isinstance(f, dict):
            f = f.get("path")
        if f:
            out.append(str(f))
    return sorted(out)

def reconstruct_with_SEAR(
    model,
    rgb_files,
    thermal_files,
    conf_quantile: float = 0.5,
    max_points: int = 200000,
    show_cameras: bool = True,
    point_source: str = "RGB + thermal",
    max_frames: int = 24,
):
    """Jointly reconstruct a 3D scene from an unpaired RGB and thermal image set.

    Args:
        rgb_files: RGB images of the scene (one trajectory).
        thermal_files: Thermal images of the same scene (a different trajectory).
        conf_quantile: Per-frame depth-confidence quantile below which points are dropped.
        max_points: Maximum number of points kept in the exported point cloud.
        show_cameras: Draw the predicted camera frusta (blue = RGB, orange = thermal).
        point_source: Which modality's points to show in the 3D view.
        max_frames: Upper bound on the total number of frames fed to the model.
    """
    t_all = time.perf_counter()
    rgb = _paths(rgb_files)
    thermal = _paths(thermal_files)

    if len(rgb) + len(thermal) < 2:
        raise RuntimeError(
            "Please provide at least 2 images in total (RGB and/or thermal)."
        )

    max_frames = int(min(max(int(max_frames), 2), MAX_FRAMES))
    if len(rgb) + len(thermal) > max_frames:
        if rgb and thermal:
            budget_rgb = max(1, round(max_frames * len(rgb) / (len(rgb) + len(thermal))))
            budget_thermal = max(1, max_frames - budget_rgb)
        else:
            budget_rgb, budget_thermal = max_frames, max_frames
        rgb = _subsample(rgb, budget_rgb)
        thermal = _subsample(thermal, budget_thermal)

    # SEAR feeds the RGB trajectory first, then the thermal one (see
    # InferenceSceneTwoTrajectories.from_scene_path in the reference repo).
    paths = rgb + thermal
    is_thermal = np.array([False] * len(rgb) + [True] * len(thermal))

    images = load_and_preprocess_images(paths)  # (S, 3, H, W), width 518, "crop" mode
    images = images[None].to("cuda")
    thermal_mask = torch.from_numpy(is_thermal)[None].to("cuda")

    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(images=images, thermal_mask=thermal_mask)

    pose_enc = predictions["pose_enc_list"][-1].to(torch.float32)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    depth = predictions["depth"][0, :, :, :, 0].to(torch.float32).cpu()
    depth_conf = predictions["depth_conf"][0].to(torch.float32).cpu()
    extrinsics = extrinsics[0].to(torch.float32).cpu()
    intrinsics = intrinsics[0].to(torch.float32).cpu()
    images_cpu = images[0].to(torch.float32).cpu()
    del predictions
    torch.cuda.empty_cache()

    return extrinsics