import pytest
from .. import fm


@pytest.mark.parametrize('nb_features, latent_dimension', [
    (10, 1),
    (100, 1),
    (100, 10),
])
def test_correct_number_coefficients(nb_features, latent_dimension):
    model = fm.build_factorization_machine(nb_features, latent_dimension)

    # Number features x size of latent vectors
    assert model.count_params() == nb_features * latent_dimension

