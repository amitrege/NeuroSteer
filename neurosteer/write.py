from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .basis import LayerBasis
from .catalog import rebuild_output, primary_tensor


def _last_positions(attention_mask: Tensor | None, hidden: Tensor) -> Tensor:
    if attention_mask is None:
        return torch.full(
            (hidden.shape[0],),
            hidden.shape[1] - 1,
            device=hidden.device,
            dtype=torch.long,
        )
    positions = attention_mask.sum(dim=1).to(hidden.device) - 1
    return positions.clamp(min=0, max=hidden.shape[1] - 1)


def _replace_selected(
    hidden: Tensor,
    replacement: Tensor,
    token: str | int | slice,
    *,
    attention_mask: Tensor | None,
) -> Tensor:
    updated = hidden.clone()
    replacement = replacement.to(device=updated.device, dtype=updated.dtype)
    if token == "all":
        return replacement
    if token == "last":
        positions = _last_positions(attention_mask, hidden)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        updated[rows, positions, :] = replacement
        return updated
    if isinstance(token, int):
        index = token if token >= 0 else hidden.shape[1] + token
        updated[:, index, :] = replacement
        return updated
    if isinstance(token, slice):
        updated[:, token, :] = replacement
        return updated
    raise ValueError(f"unsupported write token setting {token!r}")


def apply_write(
    output: Any,
    *,
    basis: LayerBasis,
    coefficients: Tensor,
    token: str | int | slice,
    operator: str,
    attention_mask: Tensor | None,
) -> Any:
    hidden = primary_tensor(output)
    basis = LayerBasis(
        components=basis.components.to(hidden.device, hidden.dtype),
        explained=basis.explained.to(hidden.device, hidden.dtype),
    )
    coefficients = coefficients.to(hidden.device, hidden.dtype)

    if coefficients.ndim == 1:
        coefficients = coefficients.unsqueeze(0)

    target = basis.compose(coefficients)
    if token == "all":
        view = hidden
        expanded_target = target.unsqueeze(1).expand(-1, hidden.shape[1], -1)
    elif token == "last":
        positions = _last_positions(attention_mask, hidden)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        view = hidden[rows, positions, :]
        expanded_target = target
    elif isinstance(token, int):
        index = token if token >= 0 else hidden.shape[1] + token
        view = hidden[:, index, :]
        expanded_target = target
    elif isinstance(token, slice):
        view = hidden[:, token, :]
        expanded_target = target.unsqueeze(1).expand(-1, view.shape[1], -1)
    else:
        raise ValueError(f"unsupported write token setting {token!r}")

    if operator == "add":
        replacement = view + expanded_target
    elif operator == "remove_projection":
        replacement = view - basis.project(view)
    elif operator == "project_subspace":
        replacement = view - basis.project(view) + expanded_target
    else:
        raise ValueError(f"unsupported write operator {operator!r}")
    replacement = replacement.to(hidden.device, hidden.dtype)

    updated = _replace_selected(
        hidden,
        replacement,
        token,
        attention_mask=attention_mask,
    )
    return rebuild_output(output, updated)
