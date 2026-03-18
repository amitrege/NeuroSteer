import pytest

torch = pytest.importorskip("torch")

from neurosteer.capture import _select_positions, read_prompt_batch
from neurosteer.pairs import ReadSpec


def test_read_prompt_batch_returns_last_nonpad_states(tiny_model, tiny_tokenizer):
    states = read_prompt_batch(
        tiny_model,
        tiny_tokenizer,
        ["abc", "abcdefgh"],
        layers=[0, 2],
        read_spec=ReadSpec(layer=0, token="last"),
        batch_size=2,
    )
    assert states[0].shape == (2, 32)
    assert states[2].shape == (2, 32)


def test_select_positions_last_clamps_when_mask_is_longer_than_hidden():
    hidden = torch.tensor([[[7.0, 1.0]]])
    attention_mask = torch.tensor([[1, 1, 1, 1]])
    selected = _select_positions(hidden, attention_mask, token="last", reduce="none")
    assert torch.allclose(selected, torch.tensor([[7.0, 1.0]]))
