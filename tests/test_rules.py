import pytest

torch = pytest.importorskip("torch")

from neurosteer.rules import ConstantRule, DiagonalGuard, LinearRule, ThresholdRule


def test_constant_rule_builds_rank_vector():
    output = ConstantRule([1.0, 0.5]).make(rank=2, batch_size=3, device=torch.device("cpu"))
    assert output.coefficients.shape == (3, 2)
    assert torch.allclose(output.coefficients[0], torch.tensor([1.0, 0.5]))


def test_threshold_and_guard_work_together():
    coords = torch.tensor([[0.2, 0.0], [2.0, 0.0]])
    rule = ThresholdRule(threshold=1.0, on_strength=1.0, off_strength=0.0, direction="below")
    output = rule.evaluate(coords, rank=2)
    guard = DiagonalGuard.fit(coords[:1], cutoff=1.0)
    allowed, distance = guard.allow(coords)

    assert output.coefficients.shape == (2, 2)
    assert allowed.tolist() == [True, False]
    assert distance.shape == (2,)


def test_linear_rule_uses_first_coordinate_by_default():
    coords = torch.tensor([[1.0, 5.0]])
    output = LinearRule(slope=2.0, bias=-1.0).evaluate(coords, rank=1)
    assert torch.allclose(output.strength, torch.tensor([1.0]))
