import numpy as np
import pandas as pd
import pytest

from risk_backtest import convalida, rischio


def test_kupiec():
    # Violazioni pari a quelle attese: nessun motivo per respingere il modello
    lr, p = convalida.test_kupiec(10, 1000, 0.99)
    assert lr == pytest.approx(0, abs=1e-9) and p == pytest.approx(1)
    # Il triplo delle violazioni attese: modello respinto
    assert convalida.test_kupiec(30, 1000, 0.99)[1] < 0.001
    assert convalida.test_kupiec(0, 1000, 0.99)[1] < 0.01


def test_christoffersen_riconosce_i_grappoli():
    sparse = np.zeros(1000, dtype=int)
    sparse[::100] = 1
    grappoli = np.zeros(1000, dtype=int)
    grappoli[500:510] = 1
    assert convalida.test_christoffersen(sparse)[1] > 0.3
    assert convalida.test_christoffersen(grappoli)[1] < 0.001


def test_semaforo_di_basilea():
    assert [convalida.semaforo_basilea(n) for n in (0, 4, 5, 9, 10)] == ["Verde", "Verde", "Giallo", "Giallo", "Rosso"]


def test_previsioni_senza_dati_futuri(prezzi, pesi):
    rp = rischio.rendimenti_portafoglio(rischio.rendimenti(prezzi), pesi).iloc[:1500]
    v = convalida.previsioni_var(rp, 0.99, finestra=500)
    # Cambiare il rendimento di un giorno non cambia le previsioni fino a quel giorno compreso
    alterato = rp.copy()
    giorno = rp.index[1200]
    alterato[giorno] = -0.2
    v2 = convalida.previsioni_var(alterato, 0.99, finestra=500)
    pd.testing.assert_frame_equal(v.loc[:giorno], v2.loc[:giorno])
    # Dal giorno dopo il VaR sale (Cornish-Fisher escluso: con un singolo valore estremo l'espansione
    # perde validità, uno dei suoi limiti noti)
    metodi = ["Storico", "Normale", "EWMA (RiskMetrics)", "GARCH(1,1)"]
    assert (v2.loc[giorno:].iloc[1][metodi] > v.loc[giorno:].iloc[1][metodi]).all()


def test_riepilogo_sui_dati_dimostrativi(prezzi, pesi):
    rp = rischio.rendimenti_portafoglio(rischio.rendimenti(prezzi), pesi)
    t = convalida.riepilogo(rp, convalida.previsioni_var(rp, 0.99), 0.99)
    # La normale sottostima le code e viene respinta; il GARCH supera entrambi i test
    assert t.loc["Normale", "p-value Kupiec"] < 0.01
    assert t.loc["GARCH(1,1)", "p-value Kupiec"] > 0.05
    assert t.loc["GARCH(1,1)", "p-value indipendenza"] > 0.05
    assert t.loc["Storico", "p-value indipendenza"] < 0.05  # violazioni a grappoli nelle crisi
