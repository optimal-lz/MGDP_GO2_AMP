"""Robot state projection shared with the motion dataset."""

from typing import List

import torch


GO2_DOF_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                 for joint in ("hip", "thigh", "calf")]


def validate_go2_dofs(names: List[str]) -> None:
    if list(names) != GO2_DOF_NAMES:
        raise ValueError(f"Go2 DOF order mismatch: expected {GO2_DOF_NAMES}, got {list(names)}")


def pack_amp_state(joint_pos, base_lin_vel, base_ang_vel, joint_vel):
    tensors = (joint_pos, base_lin_vel, base_ang_vel, joint_vel)
    if any(x.ndim != 2 for x in tensors) or [x.shape[1] for x in tensors] != [12, 3, 3, 12]:
        raise ValueError("AMP state needs 12+3+3+12 features")
    if len({x.shape[0] for x in tensors}) != 1:
        raise ValueError("AMP state batch sizes differ")
    state = torch.cat(tensors, dim=-1)
    if state.shape[1] != 30:
        raise ValueError("AMP state width must be 30")
    if not torch.isfinite(state).all():
        raise ValueError("AMP state contains non-finite values")
    return state
