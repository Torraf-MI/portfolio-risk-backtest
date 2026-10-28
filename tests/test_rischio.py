import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_backtest import rischio


def test_var_e_cvar_storici_su_un_caso_noto():
    # Perdite da 1% a 100%: al 95% il VaR è il quantile, il CVaR la media delle 5 perdite peggiori
    r = -np.arange(1, 101) / 100
    assert rischio.var_storico(r, 0.95) == pytest.approx(0.9505, abs=1e-4)
    assert rischio.cvar_storico(r, 0.95) == pytest.approx(np.mean([0.96, 0.97, 0.98, 0.99, 1.00]))


def test_var_normale_formula_chiusa():
    assert rischio.var_normale(0.0, 0.01, 0.99) == pytest.approx(0.02326, abs=1e-5)
    # Il CVaR normale al 97,5% coincide quasi con il VaR al 99% (motivo della scelta di Basilea)
    assert rischio.cvar_normale(0, 0.01, 0.975) == pytest.approx(rischio.var_normale(0, 0.01, 0.99), rel=0.01)


def test_cornish_fisher_si_riduce_alla_normale_senza_asimmetria_e_curtosi():
    assert rischio.quantile_cornish_fisher(0.01, 0, 0) == pytest.approx(stats.norm.ppf(0.01))
    # Asimmetria negativa e code pesanti spostano il quantile più in basso
    assert rischio.quantile_cornish_fisher(0.01, -0.5, 5) < stats.norm.ppf(0.01)


def test_t_di_student_riconosce_le_code_pesanti():
    x = stats.t.rvs(4, scale=0.01, size=20_000, random_state=1)
    var, cvar, parametri = rischio.var_t(x, 0.99)
    assert parametri["gradi_liberta"] == pytest.approx(4, abs=0.6)
    assert var == pytest.approx(-0.01 * stats.t.ppf(0.01, 4), rel=0.05)
    assert cvar > var


def test_ewma_ricorsione():
    r = pd.Series([0.01, -0.02, 0.005, 0.0, 0.03])
    lam, var = 0.94, np.mean(r.to_numpy() ** 2)
    attese = []
    for x in r:
        var = lam * var + (1 - lam) * x**2
        attese.append(np.sqrt(var))
    np.testing.assert_allclose(rischio.volatilita_ewma(r, lam), attese)


def test_garch_recupera_i_parametri():
    rng = np.random.default_rng(0)
    omega, alpha, beta, n = 2e-6, 0.08, 0.90, 8000
    e, s2 = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        e[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * e[t] ** 2 + beta * s2
    g = rischio.stima_garch(e)
    assert g.alpha == pytest.approx(alpha, abs=0.03)
    assert g.beta == pytest.approx(beta, abs=0.04)
    assert g.volatilita_lungo_periodo == pytest.approx(np.sqrt(omega / (1 - alpha - beta) * 252), rel=0.15)


def test_tabella_var_e_radice_del_tempo(prezzi, pesi):
    rp = rischio.rendimenti_portafoglio(rischio.rendimenti(prezzi), pesi)
    t1 = rischio.tabella_var(rp, 0.99, 1)
    t21 = rischio.tabella_var(rp, 0.99, 21)
    assert list(t1.index) == rischio.METODI
    assert (t1["CVaR"] >= t1["VaR"]).all()
    np.testing.assert_allclose(t21["VaR"], t1["VaR"] * np.sqrt(21))
    # Le code dei dati sono più pesanti della normale
    assert t1.loc["Storico", "VaR"] > t1.loc["Normale", "VaR"]


def test_i_contributi_sommano_al_totale(prezzi, pesi):
    r = rischio.rendimenti(prezzi)
    rp = rischio.rendimenti_portafoglio(r, pesi)
    assert rischio.contributi_volatilita(r, pesi).sum() == pytest.approx(rp.std() * np.sqrt(252))
    assert rischio.contributi_cvar(r, pesi, 0.99).sum() == pytest.approx(rischio.cvar_storico(rp, 0.99))
