import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, padding=True, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]

        encoded = []
        for text in texts:
            row = [((ord(ch) % 23) + 2) for ch in text[:18]]
            encoded.append(row or [2])

        width = max(len(row) for row in encoded)
        ids = []
        mask = []
        for row in encoded:
            pad = width - len(row)
            ids.append(row + ([self.pad_token_id] * pad))
            mask.append(([1] * len(row)) + ([0] * pad))

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


@pytest.fixture
def tiny_model():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


@pytest.fixture
def tiny_tokenizer():
    return TinyTokenizer()
