import pytest

torch = pytest.importorskip("torch")

from neurosteer.basis import Basis


def test_fit_mean_delta_and_pca_rank_shapes():
    positive = {0: torch.tensor([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])}
    negative = {0: torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])}

    mean_basis = Basis.fit_from_states(positive, negative, method="mean_delta")
    pca_basis = Basis.fit_from_states(positive, negative, method="pca_k", rank=1)

    assert mean_basis.layers[0].components.shape == (1, 2)
    assert pca_basis.layers[0].components.shape == (1, 2)
    assert pca_basis.layers[0].explained.shape == (1,)
