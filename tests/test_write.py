import pytest

torch = pytest.importorskip("torch")

from neurosteer.basis import LayerBasis
from neurosteer.write import apply_write


def test_apply_write_remove_projection_changes_selected_slice_only():
    basis = LayerBasis(components=torch.tensor([[1.0, 0.0]]))
    output = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    updated = apply_write(
        output,
        basis=basis,
        coefficients=torch.tensor([[0.0]]),
        token="last",
        operator="remove_projection",
        attention_mask=torch.tensor([[1, 1]]),
    )
    assert torch.allclose(updated[:, 0, :], output[:, 0, :])
    assert torch.allclose(updated[:, 1, :], torch.tensor([[0.0, 4.0]]))


def test_apply_write_preserves_hidden_dtype_with_float32_coefficients():
    basis = LayerBasis(components=torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    output = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float16)
    updated = apply_write(
        output,
        basis=basis,
        coefficients=torch.tensor([[0.5]], dtype=torch.float32),
        token="last",
        operator="add",
        attention_mask=torch.tensor([[1, 1]]),
    )
    assert updated.dtype == torch.float16
    assert torch.allclose(updated[:, 0, :], output[:, 0, :])
    assert torch.allclose(updated[:, 1, :], torch.tensor([[3.5, 4.0]], dtype=torch.float16))
