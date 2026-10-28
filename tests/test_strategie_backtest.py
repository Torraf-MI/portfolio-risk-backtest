import numpy as np
import pandas as pd
import pytest

from risk_backtest import backtest, metriche, strategie
from risk_backtest.dati import classe


def test_classi_riconosciute(prezzi):
    assert [classe(c) for c in prezzi.columns] == [
        "azioni", "azioni", "azioni", "governativi", "corporate", "oro", "liquidita"]
    assert classe("VWCE.DE") == "azioni" and classe("XEON.DE") == "liquidita"


def test_risk_parity_eguaglia_i_contributi(prezzi):
    w = strategie.risk_parity(prezzi.loc[:"2015"])
    rischiosi = w[w > 0].index
    cov = strategie.covarianza(prezzi.loc[:"2015", rischiosi]).to_numpy()
    x = w[rischiosi].to_numpy()
    contributi = x * (cov @ x)
    np.testing.assert_allclose(contributi / contributi.sum(), 1 / len(x), atol=1e-4)
    assert w.sum() == pytest.approx(1)


def test_minima_varianza_meno_rischiosa_dei_pesi_uguali(prezzi):
    p = prezzi.loc[:"2015"]
    cov = strategie.covarianza(p).to_numpy()
    mv, eq = strategie.minima_varianza(p).to_numpy(), strategie.pesi_uguali(p).to_numpy()
    assert mv @ cov @ mv < eq @ cov @ eq
    assert mv.max() <= 0.4 + 1e-6


def test_filtro_di_tendenza_sposta_sul_monetario(prezzi):
    p = prezzi.loc[:"2008-12-31"]  # in piena crisi quasi tutte le azioni sono sotto la media mobile
    base = strategie.bilanciato_60_40(p)
    w = strategie.trend_following()(p)
    sotto = p.iloc[-1] < p.iloc[-200:].mean()
    assert sotto[[c for c in p if classe(c) == "azioni"]].sum() >= 2
    assert (w[sotto] == 0).all()
    assert w["Monetario EUR"] == pytest.approx(base[sotto].sum())
    assert w.sum() == pytest.approx(1)


def test_volatilita_obiettivo_non_usa_leva(prezzi):
    for data in ["2008-11-28", "2017-06-30"]:
        w = strategie.volatilita_obiettivo()(prezzi.loc[:data])
        assert w.sum() == pytest.approx(1) and (w >= 0).all()
    assert strategie.volatilita_obiettivo()(prezzi.loc[:"2008-11-28"])["Monetario EUR"] > 0.3


def test_backtest_a_pesi_costanti_senza_costi_coincide_col_calcolo_manuale():
    date = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(0)
    prezzi = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (600, 2)), axis=0)), index=date,
                          columns=["A", "B"])
    w = pd.Series({"A": 0.6, "B": 0.4})
    ris = backtest.esegui(prezzi, strategie.costante(w), costi_bps=0, frequenza="YE")
    # Verifica manuale: deriva dei pesi da un ribilanciamento annuale all'altro
    r = prezzi.pct_change().loc[ris.rendimenti.index]
    valore, quote = 1.0, None
    attesi = []
    fine_anno = set(backtest.date_ribilanciamento(prezzi.index[252:], "YE"))
    for d in r.index:
        if quote is None:
            quote = w.to_numpy() * valore
        quote = quote * (1 + r.loc[d].to_numpy())
        nuovo = quote.sum()
        attesi.append(nuovo / valore - 1)
        valore = nuovo
        if d in fine_anno:
            quote = w.to_numpy() * valore
    np.testing.assert_allclose(ris.rendimenti, attesi, atol=1e-12)


def test_i_costi_riducono_il_rendimento(prezzi):
    s = strategie.trend_following()
    senza = backtest.esegui(prezzi.loc[:"2014"], s, costi_bps=0)
    con = backtest.esegui(prezzi.loc[:"2014"], s, costi_bps=50)
    differenza = (1 + senza.rendimenti).prod() - (1 + con.rendimenti).prod()
    assert differenza > 0
    assert con.rotazione.iloc[0] == pytest.approx(1.0)  # acquisto iniziale dell'intero portafoglio


def test_metriche_su_casi_noti():
    date = pd.bdate_range("2020-01-01", periods=504)
    r = pd.Series(0.0, index=date)
    r.iloc[10], r.iloc[20] = -0.5, 1.0  # dimezza e poi raddoppia: torna al punto di partenza
    assert metriche.drawdown(r).min() == pytest.approx(-0.5)
    assert metriche.cagr(r) == pytest.approx(0, abs=1e-12)
    assert metriche.durata_massima_drawdown(r) == 10
    crescita = pd.Series((1.10) ** (1 / 252) - 1, index=date)
    assert metriche.cagr(crescita) == pytest.approx(0.10)


def test_sharpe_probabilistico():
    rng = np.random.default_rng(1)
    date = pd.bdate_range("2000-01-01", periods=5000)
    buono = pd.Series(rng.normal(0.0006, 0.01, 5000), index=date)
    nullo = pd.Series(rng.normal(0.0, 0.01, 5000), index=date)
    assert metriche.sharpe_probabilistico(buono) > 0.99
    assert 0.02 < metriche.sharpe_probabilistico(nullo - nullo.mean()) < 0.98
    assert metriche.sharpe_probabilistico(buono, soglia=metriche.sharpe(buono)) == pytest.approx(0.5, abs=0.01)


def test_alfa_beta_di_una_strategia_a_leva():
    rng = np.random.default_rng(2)
    date = pd.bdate_range("2010-01-01", periods=2520)
    b = pd.Series(rng.normal(0.0003, 0.008, len(date)), index=date)
    ab = metriche.alfa_beta(0.5 * b, b)
    assert ab["Beta"] == pytest.approx(0.5, abs=0.01)
    assert ab["R²"] == pytest.approx(1, abs=1e-3)
