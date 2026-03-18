import pytest

torch = pytest.importorskip("torch")

from neurosteer.catalog import locate_decoder_stack


def test_locate_decoder_stack_finds_llama_layers(tiny_model):
    stack = locate_decoder_stack(tiny_model)
    assert stack.path == "model.layers"
    assert stack.count() == 3
    assert stack.normalize(-1) == 2
