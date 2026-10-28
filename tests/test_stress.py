import numpy as np
import pandas as pd
import pytest

from risk_backtest import dati, stress


def test_le_crisi_dimostrative_hanno_i_rendimenti_indicati(prezzi):
    for ep in dati.EPISODI:
        finestra = prezzi.loc[ep.inizio:ep.fine]
        base = prezzi.loc[:ep.inizio].iloc[-2] if prezzi.index[0] < pd.Timestamp(ep.inizio) else finestra.iloc[0]
        osservato = finestra.iloc[-1] / base - 1
        for asset, atteso in ep.rendimenti.items():
            assert osservato[asset] == pytest.approx(atteso, abs=0.03), (ep.nome, asset)


def test_stress_storico_senza_ribilanciamento(prezzi, pesi):
    ep = dati.EPISODI[0]
    e = stress.stress_storico(prezzi, pesi, ep)
    base = prezzi.loc[:ep.inizio].iloc[-2]
    finale = prezzi.loc[:ep.fine].iloc[-1]
    assert e.rendimento == pytest.approx(float((finale / base * pesi).sum() - 1))
    assert e.contributi.sum() == pytest.approx(e.rendimento)
    assert e.perdita_massima <= min(e.rendimento, 0)


def test_crisi_fuori_dai_dati_ignorata(prezzi, pesi):
    assert stress.stress_storico(prezzi.loc["2015":], pesi, dati.EPISODI[0]) is None


def test_scenario_ipotetico_propaga_lo_shock_con_la_regressione():
    rng = np.random.default_rng(3)
    date = pd.bdate_range("2010-01-01", periods=2500)
    azioni = rng.normal(0, 0.01, len(date))
    rumore = rng.normal(0, 0.001, len(date))
    rendimenti = pd.DataFrame({"Azionario": azioni, "Bilanciato": 0.5 * azioni + rumore}, index=date)
    prezzi = 100 * (1 + rendimenti).cumprod()
    pesi = pd.Series({"Azionario": 0.5, "Bilanciato": 0.5})
    e = stress.stress_ipotetico(prezzi, pesi, stress.Scenario("Crollo", {"azioni": -0.30}))
    assert e.rendimenti_asset["Azionario"] == -0.30
    assert e.rendimenti_asset["Bilanciato"] == pytest.approx(-0.15, abs=0.01)
    assert e.rendimento == pytest.approx(-0.225, abs=0.01)


def test_peggiori_periodi_non_sovrapposti(prezzi, pesi):
    rp = (prezzi.pct_change().dropna() @ pesi)
    t = stress.peggiori_periodi(rp, 21, 3)
    assert len(t) == 3 and t["Rendimento"].is_monotonic_increasing
    fini = sorted(t["Fine"])
    assert all((b - a).days > 21 for a, b in zip(fini, fini[1:]))
