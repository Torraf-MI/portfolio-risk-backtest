"""Riga di comando: report di rischio e backtest per il portafoglio di un cliente.

Esempi::

    risk-backtest --demo --cliente "Mario Rossi" --importo 500000
    risk-backtest --csv data/prezzi.csv --pesi "Azionario=0.6, Obbligazioni=0.4" --livello 0.95
    risk-backtest --ticker VWCE.DE XGLE.DE SGLD.MI --pesi "VWCE.DE=0.6, XGLE.DE=0.3, SGLD.MI=0.1"
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import dati
from .report import _eur, analizza, build_report, salva_grafici
from .stress import tabella_stress


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="risk-backtest", description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV di prezzi: colonna data + una colonna per asset")
    src.add_argument("--ticker", nargs="+", help="Ticker Yahoo Finance da scaricare")
    src.add_argument("--demo", action="store_true", help="Usa dati sintetici dimostrativi (2007-2025)")
    p.add_argument("--da", default="2007-01-01", help="Data iniziale per --ticker (AAAA-MM-GG)")
    p.add_argument("--pesi", help='Pesi del portafoglio, ad esempio "Azionario Globale=0.6, Oro=0.4" '
                                  "(default: esempio bilanciato con --demo, pesi uguali altrimenti)")
    p.add_argument("--importo", type=float, default=100_000, help="Controvalore del portafoglio in euro")
    p.add_argument("--livello", type=float, default=0.99, help="Livello di confidenza del VaR (default 0.99)")
    p.add_argument("--orizzonte", type=int, default=21, help="Orizzonte del VaR in giorni di borsa (default 21)")
    p.add_argument("--risk-free", type=float, default=0.015, help="Tasso privo di rischio annuo")
    p.add_argument("--costi-bps", type=float, default=10.0, help="Costi di transazione in punti base")
    p.add_argument("--inizio-backtest", help="Prima data del backtest delle strategie (AAAA-MM-GG)")
    p.add_argument("--senza-strategie", action="store_true", help="Solo analisi del rischio, senza confronto")
    p.add_argument("--cliente", default="Cliente", help="Nome del cliente nel report")
    p.add_argument("--output", default="output/report.html", help="Percorso del report HTML")
    p.add_argument("--grafici", help="Cartella dove salvare anche i grafici PNG")
    p.add_argument("--offline", action="store_true", help="Include plotly.js nel report (funziona senza rete)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    note = ""
    if args.csv:
        prezzi = dati.load_csv(args.csv)
    elif args.ticker:
        prezzi = dati.download_prices(args.ticker, start=args.da)
    else:
        prezzi = dati.generate_sample_prices()
        note = "Analisi svolta su dati sintetici dimostrativi, non su strumenti reali."

    colonne = list(prezzi.columns)
    if args.pesi:
        pesi = dati.parse_pesi(args.pesi, colonne)
    elif args.demo:
        pesi = dati.normalizza_pesi(dati.PESI_ESEMPIO, colonne)
    else:
        pesi = pd.Series(1 / len(colonne), index=colonne)

    a = analizza(prezzi, pesi, importo=args.importo, livello=args.livello, orizzonte=args.orizzonte,
                 risk_free=args.risk_free, costi_bps=args.costi_bps, inizio_backtest=args.inizio_backtest,
                 con_strategie=not args.senza_strategie)
    path = build_report(a, args.output, client_name=args.cliente, data_note=note, offline=args.offline)
    if args.grafici:
        salva_grafici(a, args.grafici)

    pd.set_option("display.width", 160)
    print(f"VaR e CVaR al {args.livello:.0%} su {args.orizzonte} giorni, portafoglio di {_eur(args.importo)}")
    print((a.var_orizzonte[["VaR", "CVaR"]] * args.importo).map(_eur).to_string())
    print("\nBacktest del VaR a un giorno")
    print(a.convalida[["Violazioni attese", "Violazioni", "p-value Kupiec", "p-value indipendenza", "Semaforo"]]
          .to_string(formatters={"Violazioni attese": "{:.0f}".format, "p-value Kupiec": "{:.3f}".format,
                                 "p-value indipendenza": "{:.3f}".format}))
    print("\nStress test")
    t = tabella_stress(a.stress, args.importo)
    print(t[["Tipo", "Rendimento", "Impatto"]].to_string(formatters={"Rendimento": "{:.1%}".format,
                                                                      "Impatto": _eur}))
    if not a.metriche.empty:
        print("\nStrategie")
        cols = ["Rendimento annuo", "Volatilità", "Sharpe", "Massima perdita"]
        print(a.metriche[cols].to_string(formatters={"Rendimento annuo": "{:.1%}".format,
                                                     "Volatilità": "{:.1%}".format, "Sharpe": "{:.2f}".format,
                                                     "Massima perdita": "{:.1%}".format}))
    print(f"\nReport salvato in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
