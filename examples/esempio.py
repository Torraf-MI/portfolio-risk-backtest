"""Uso del progetto come libreria: rischio, backtest del VaR, stress test e confronto tra strategie."""

from risk_backtest import (
    confronta,
    generate_sample_prices,
    normalizza_pesi,
    previsioni_var,
    rendimenti,
    rendimenti_portafoglio,
    riepilogo,
    stima_garch,
    strategie_standard,
    stress_test,
    tabella_var,
)
from risk_backtest.dati import PESI_ESEMPIO
from risk_backtest.metriche import tabella
from risk_backtest.stress import tabella_stress

prezzi = generate_sample_prices()  # oppure load_csv("prezzi.csv") o download_prices([...])
pesi = normalizza_pesi(PESI_ESEMPIO, list(prezzi.columns))
importo = 500_000

rp = rendimenti_portafoglio(rendimenti(prezzi), pesi)

# 1. Quanto si può perdere in un mese, con sei metodi
print((tabella_var(rp, livello=0.99, orizzonte=21)[["VaR", "CVaR"]] * importo).round(0))

# 2. Il rischio di oggi secondo il GARCH(1,1)
g = stima_garch(rp)
print(f"GARCH: alpha {g.alpha:.3f}, beta {g.beta:.3f}, volatilità di lungo periodo {g.volatilita_lungo_periodo:.1%}")

# 3. I modelli avrebbero funzionato? Previsioni fuori campione e test statistici
print(riepilogo(rp, previsioni_var(rp, 0.99), 0.99)[["Violazioni attese", "Violazioni", "p-value Kupiec",
                                                    "p-value indipendenza", "Semaforo"]])

# 4. Crisi storiche e scenari ipotetici sul portafoglio di oggi
print(tabella_stress(stress_test(prezzi, pesi), importo)[["Tipo", "Rendimento", "Impatto"]].round(3))

# 5. Portafoglio attuale contro 60/40, risk parity, minima varianza, volatilità obiettivo, filtro di tendenza
risultati = confronta(prezzi, strategie_standard(pesi), costi_bps=10)
print(tabella({nome: r.rendimenti for nome, r in risultati.items()}, risk_free=0.015)[
    ["Rendimento annuo", "Volatilità", "Sharpe", "Massima perdita"]].round(3))
