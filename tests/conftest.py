import pytest

from risk_backtest.dati import PESI_ESEMPIO, generate_sample_prices, normalizza_pesi


@pytest.fixture(scope="session")
def prezzi():
    return generate_sample_prices()


@pytest.fixture(scope="session")
def pesi(prezzi):
    return normalizza_pesi(PESI_ESEMPIO, list(prezzi.columns))
