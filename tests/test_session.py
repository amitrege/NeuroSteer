import pytest

torch = pytest.importorskip("torch")

from neurosteer import Basis, ConstantRule, PromptPair, ReadSpec, Session, ThresholdRule, WriteSpec


def _demo_basis(layer: int, hidden: int):
    positive = {layer: torch.ones(4, hidden)}
    negative = {layer: torch.zeros(4, hidden)}
    return Basis.fit_from_states(positive, negative, method="mean_delta")


def test_static_session_changes_logits_and_cleans_up(tiny_model, tiny_tokenizer):
    encoded = tiny_tokenizer(["hello world"], return_tensors="pt")
    basis = _demo_basis(layer=1, hidden=tiny_model.config.hidden_size)
    session = Session(
        model=tiny_model,
        write_basis=basis,
        write_spec=WriteSpec.from_layers([1], token="last", operator="add"),
        rule=ConstantRule(2.0),
        mode="static",
    )

    baseline = tiny_model(**encoded).logits.detach()
    with session:
        changed = tiny_model(**encoded).logits.detach()
    restored = tiny_model(**encoded).logits.detach()

    assert not torch.allclose(baseline, changed)
    assert torch.allclose(baseline, restored)


def test_same_pass_rejects_noncausal_setup(tiny_model):
    basis = _demo_basis(layer=2, hidden=tiny_model.config.hidden_size)
    with pytest.raises(ValueError):
        Session(
            model=tiny_model,
            write_basis=basis,
            write_spec=WriteSpec.from_layers([1], token="last"),
            read_basis=basis,
            read_spec=ReadSpec(layer=2, token="last"),
            rule=ThresholdRule(threshold=0.0),
            mode="same_pass",
        )


def test_next_token_mode_uses_queued_coefficients(tiny_model, tiny_tokenizer):
    encoded = tiny_tokenizer(["prompt"], return_tensors="pt")
    read_basis = _demo_basis(layer=0, hidden=tiny_model.config.hidden_size)
    write_basis = _demo_basis(layer=1, hidden=tiny_model.config.hidden_size)
    session = Session(
        model=tiny_model,
        write_basis=write_basis,
        write_spec=WriteSpec.from_layers([1], token="last", operator="add"),
        read_basis=read_basis,
        read_spec=ReadSpec(layer=0, token="last"),
        rule=ThresholdRule(threshold=10.0, on_strength=1.0, off_strength=0.0, direction="below"),
        mode="next_token",
    )

    baseline = tiny_model(**encoded).logits.detach()
    with session:
        first = tiny_model(**encoded).logits.detach()
        second = tiny_model(**encoded).logits.detach()

    assert torch.allclose(baseline, first)
    assert not torch.allclose(baseline, second)


def test_preview_call_runs_two_pass_path(tiny_model, tiny_tokenizer):
    encoded = tiny_tokenizer(["debug"], return_tensors="pt")
    read_basis = _demo_basis(layer=2, hidden=tiny_model.config.hidden_size)
    write_basis = _demo_basis(layer=0, hidden=tiny_model.config.hidden_size)
    session = Session(
        model=tiny_model,
        write_basis=write_basis,
        write_spec=WriteSpec.from_layers([0], token="last", operator="add"),
        read_basis=read_basis,
        read_spec=ReadSpec(layer=2, token="last"),
        rule=ThresholdRule(threshold=10.0, on_strength=1.0, off_strength=0.0, direction="below"),
        mode="preview",
    )

    output = session.preview_call(**encoded)
    assert output.logits.shape[0] == 1
    phases = [event.phase for event in session.trace.events]
    pass_names = [event.pass_name for event in session.trace.events]
    assert "read" in phases
    assert "write" in phases
    assert "preview" in pass_names
