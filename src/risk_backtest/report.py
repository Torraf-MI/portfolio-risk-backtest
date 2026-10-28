"""Analisi completa, grafici interattivi (plotly) e report HTML per il cliente."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape
from plotly.offline import get_plotlyjs, get_plotlyjs_version
from scipy import stats

from . import backtest, convalida, metriche, rischio, strategie
from .dati import classe
from .stress import EsitoStress, peggiori_periodi, stress_test, tabella_stress

BLU, ARANCIO, ACQUA, VIOLA = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
BLU_SCURO, ROSSO, GIALLO, GRIGIO = "#1c5cab", "#e34948", "#e8a317", "#8c8b86"
TESTO, TESTO_SECONDARIO, GRIGLIA = "#222222", "#52514e", "#e6e5e1"
COLORI = [BLU_SCURO, ARANCIO, ACQUA, VIOLA, GIALLO, "#d14d9c", GRIGIO, BLU]
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"


def _layout(fig: go.Figure, titolo: str, altezza: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": titolo, "x": 0, "font": {"size": 15, "color": TESTO}},
        height=altezza,
        template="plotly_white",
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 12, "color": TESTO_SECONDARIO},
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        separators=",.",
        hoverlabel={"bgcolor": "white", "font_color": TESTO},
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    fig.update_xaxes(gridcolor=GRIGLIA, zeroline=False)
    fig.update_yaxes(gridcolor=GRIGLIA, zerolinecolor="#bdbcb6")
    return fig


@dataclass
class Analisi:
    """Risultato completo dell'analisi, usato per grafici e report."""

    prezzi: pd.DataFrame
    pesi: pd.Series
    importo: float
    livello: float
    orizzonte: int
    risk_free: float
    costi_bps: float
    rendimenti: pd.Series  # rendimenti giornalieri del portafoglio
    var: pd.DataFrame  # VaR e CVaR per metodo a un giorno
    var_orizzonte: pd.DataFrame  # VaR e CVaR per metodo all'orizzonte
    garch: rischio.Garch
    parametri_t: dict[str, float]
    previsioni: pd.DataFrame  # VaR previsto giorno per giorno
    convalida: pd.DataFrame
    stress: list[EsitoStress]
    peggiori_mesi: pd.DataFrame
    contributi: pd.DataFrame
    correlazione: pd.Series | None
    backtest: dict[str, backtest.RisultatoBacktest] = field(repr=False)
    metriche: pd.DataFrame = field(repr=False)
    confronto: pd.DataFrame = field(repr=False)  # alfa e beta rispetto al 60/40
    annuali: pd.DataFrame = field(repr=False)


def _correlazione_azioni_obbligazioni(r: pd.DataFrame) -> pd.Series | None:
    azioni = [c for c in r if classe(c) == "azioni"]
    gov = [c for c in r if classe(c) == "governativi"]
    if not azioni or not gov:
        return None
    return rischio.correlazione_mobile(r, azioni[0], gov[0]).rename("Correlazione azioni-obbligazioni")


def analizza(
    prezzi: pd.DataFrame,
    pesi: pd.Series,
    importo: float = 100_000,
    livello: float = 0.99,
    orizzonte: int = 21,
    risk_free: float = 0.015,
    costi_bps: float = 10.0,
    inizio_backtest: str | None = None,
    con_strategie: bool = True,
) -> Analisi:
    """Esegue tutte le analisi del report sul portafoglio indicato."""
    r = rischio.rendimenti(prezzi)
    rp = rischio.rendimenti_portafoglio(r, pesi)
    x = rp.to_numpy()
    _, _, parametri_t = rischio.var_t(x, livello)
    previsioni = convalida.previsioni_var(rp, livello)
    contributi = pd.DataFrame({
        "Peso": pesi,
        "Contributo alla volatilità": rischio.contributi_volatilita(r, pesi),
        "Contributo al CVaR": rischio.contributi_cvar(r, pesi, livello),
    })
    contributi["Quota della volatilità"] = contributi["Contributo alla volatilità"] / contributi[
        "Contributo alla volatilità"].sum()
    contributi["Quota del CVaR"] = contributi["Contributo al CVaR"] / contributi["Contributo al CVaR"].sum()

    if con_strategie:
        ris = backtest.confronta(prezzi, strategie.strategie_standard(pesi), inizio=inizio_backtest,
                                 costi_bps=costi_bps)
        serie = {k: v.rendimenti for k, v in ris.items()}
        tab = metriche.tabella(serie, risk_free)
        tab["Rotazione annua"] = [v.rotazione_annua for v in ris.values()]
        confronto = pd.DataFrame({k: metriche.alfa_beta(s, serie["60/40"], risk_free)
                                  for k, s in serie.items() if k != "60/40"}).T
        annuali = metriche.rendimenti_annuali(serie)
    else:
        ris, tab, confronto, annuali = {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    return Analisi(
        prezzi=prezzi,
        pesi=pesi,
        importo=importo,
        livello=livello,
        orizzonte=orizzonte,
        risk_free=risk_free,
        costi_bps=costi_bps,
        rendimenti=rp,
        var=rischio.tabella_var(rp, livello, 1),
        var_orizzonte=rischio.tabella_var(rp, livello, orizzonte),
        garch=rischio.stima_garch(x),
        parametri_t=parametri_t,
        previsioni=previsioni,
        convalida=convalida.riepilogo(rp, previsioni, livello),
        stress=stress_test(prezzi, pesi),
        peggiori_mesi=peggiori_periodi(rp, 21, 3),
        contributi=contributi,
        correlazione=_correlazione_azioni_obbligazioni(r),
        backtest=ris,
        metriche=tab,
        confronto=confronto,
        annuali=annuali,
    )


# --------------------------------------------------------------------------- #
# Grafici
# --------------------------------------------------------------------------- #
def grafico_distribuzione(a: Analisi) -> go.Figure:
    """Istogramma dei rendimenti giornalieri con la curva normale, il VaR e il CVaR storici."""
    x = a.rendimenti.to_numpy().round(5)
    var, cvar = a.var.loc["Storico", "VaR"], a.var.loc["Storico", "CVaR"]
    fig = go.Figure()
    fig.add_histogram(x=x, nbinsx=160, histnorm="probability density", name="Rendimenti giornalieri",
                      marker={"color": BLU, "line": {"width": 0}}, opacity=0.75,
                      hovertemplate="%{x:.2%}<extra></extra>")
    griglia = np.linspace(x.min(), x.max(), 400)
    fig.add_scatter(x=griglia, y=stats.norm.pdf(griglia, x.mean(), x.std()), mode="lines",
                    name="Distribuzione normale con stessa media e volatilità", line={"color": GRIGIO, "dash": "dot"})
    for valore, nome, colore in [(var, f"VaR {a.livello:.0%}", ARANCIO), (cvar, f"CVaR {a.livello:.0%}", ROSSO)]:
        fig.add_vline(x=-valore, line={"color": colore, "width": 2})
        fig.add_annotation(x=-valore, y=1, yref="paper", text=f"{nome}<br>{_pct(-valore, 2)}", showarrow=False,
                           xanchor="right", yanchor="top", font={"color": colore}, xshift=-4)
    fig.update_xaxes(tickformat=".1%", range=[min(-cvar * 1.6, x.min() * 0.6), -x.min() * 0.5],
                     title="Rendimento giornaliero del portafoglio")
    fig.update_yaxes(title="Densità", showticklabels=False)
    fig.update_layout(bargap=0)
    return _layout(fig, "Distribuzione dei rendimenti giornalieri: le code sono più pesanti della normale")


def grafico_metodi(a: Analisi) -> go.Figure:
    """VaR e CVaR in euro all'orizzonte, metodo per metodo."""
    t = a.var_orizzonte * a.importo
    fig = go.Figure()
    fig.add_bar(x=t.index, y=t["VaR"], name=f"VaR {a.livello:.0%}", marker_color=BLU_SCURO,
                text=[_eur(v) for v in t["VaR"]], textposition="outside", hovertemplate="%{x}<br>VaR € %{y:,.0f}")
    fig.add_bar(x=t.index, y=t["CVaR"], name=f"CVaR {a.livello:.0%} (perdita media oltre il VaR)", marker_color=ARANCIO,
                text=[_eur(v) for v in t["CVaR"]], textposition="outside", hovertemplate="%{x}<br>CVaR € %{y:,.0f}")
    fig.update_yaxes(tickprefix="€ ", tickformat=",.0f", range=[0, t["CVaR"].max() * 1.18])
    fig.update_layout(barmode="group", bargroupgap=0.08)
    return _layout(fig, f"Perdita potenziale su {_orizzonte(a.orizzonte)} con sei metodi", altezza=440)


def grafico_convalida(a: Analisi) -> go.Figure:
    """Perdite giornaliere e VaR previsto il giorno prima; i punti rossi sono le violazioni."""
    perdite = (-a.rendimenti.loc[a.previsioni.index]).round(5)
    previsioni = a.previsioni.round(5)
    fig = go.Figure()
    fig.add_scatter(x=perdite.index, y=perdite, mode="markers", name="Perdita del giorno",
                    marker={"size": 3, "color": "rgba(82,81,78,0.35)"}, hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2%}")
    for metodo, colore in [("Storico", ARANCIO), ("GARCH(1,1)", BLU_SCURO)]:
        fig.add_scatter(x=previsioni.index, y=previsioni[metodo], mode="lines", name=f"VaR {metodo}",
                        line={"color": colore, "width": 1.6}, hovertemplate="%{x|%d/%m/%Y}<br>VaR %{y:.2%}")
    v = convalida.violazioni(a.rendimenti, a.previsioni)["GARCH(1,1)"]
    fig.add_scatter(x=perdite[v].index, y=perdite[v], mode="markers", name="Violazioni del VaR GARCH",
                    marker={"size": 6, "color": ROSSO, "symbol": "x"}, hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2%}")
    fig.update_yaxes(tickformat=".1%", range=[0, perdite.max() * 1.05], title="Perdita giornaliera")
    return _layout(fig, "Backtest del VaR: il modello GARCH si adatta alle fasi di mercato", altezza=440)


def grafico_stress(a: Analisi) -> go.Figure:
    """Impatto in euro di ogni scenario di stress sul portafoglio."""
    t = tabella_stress(a.stress, a.importo).iloc[::-1]
    colori = [BLU_SCURO if tipo == "Storico" else ARANCIO for tipo in t["Tipo"]]
    fig = go.Figure(go.Bar(
        y=[f"{n} " for n in t.index], x=t["Impatto"], orientation="h", marker_color=colori,
        text=[f"{_eur(v)} ({_pct(p)})" for v, p in zip(t["Impatto"], t["Rendimento"])], textposition="outside",
        hovertemplate="%{y}<br>€ %{x:,.0f}<extra></extra>",
    ))
    minimo = t["Impatto"].min()
    fig.update_xaxes(ticksuffix=" €", tickformat=",.0f", range=[minimo * 1.4, max(0, t["Impatto"].max()) * 1.2])
    fig.add_annotation(x=0, y=1.06, xref="paper", yref="paper", showarrow=False, xanchor="left",
                       text=f"<span style='color:{BLU_SCURO}'>■</span> crisi storiche   "
                            f"<span style='color:{ARANCIO}'>■</span> scenari ipotetici")
    fig.update_layout(showlegend=False, margin={"l": 210})
    return _layout(fig, "Stress test: cosa succederebbe al portafoglio di oggi", altezza=80 + 46 * len(t))


def grafico_contributi(a: Analisi) -> go.Figure:
    """Peso in portafoglio e quota del rischio di ogni asset."""
    c = a.contributi.sort_values("Peso")
    fig = go.Figure()
    for col, nome, colore in [("Peso", "Peso nel portafoglio", GRIGIO),
                              ("Quota della volatilità", "Quota della volatilità", BLU_SCURO),
                              ("Quota del CVaR", f"Quota del CVaR {a.livello:.0%}", ARANCIO)]:
        fig.add_bar(y=c.index, x=c[col], name=nome, orientation="h", marker_color=colore,
                    hovertemplate="%{y}<br>%{x:.1%}<extra>" + nome + "</extra>")
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(barmode="group", bargap=0.25, margin={"l": 190})
    return _layout(fig, "Da dove viene il rischio: peso e contributo di ogni asset", altezza=480)


def grafico_strategie(a: Analisi) -> go.Figure:
    """Valore di 100 € investiti in ogni strategia, in scala logaritmica."""
    fig = go.Figure()
    for i, (nome, ris) in enumerate(a.backtest.items()):
        v = (ris.valore * 100).resample("W-FRI").last().round(2)  # settimanale: file più leggero
        fig.add_scatter(x=v.index, y=v, mode="lines", name=nome,
                        line={"color": COLORI[i % len(COLORI)], "width": 2.6 if i == 0 else 1.6},
                        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f}<extra>" + nome + "</extra>")
    fig.update_yaxes(type="log", title="Valore di 100 € investiti")
    fig.update_layout(hovermode="x unified")
    return _layout(fig, "Backtest delle strategie al netto dei costi", altezza=460)


def grafico_drawdown(a: Analisi) -> go.Figure:
    """Perdita dal massimo precedente per ogni strategia."""
    fig = go.Figure()
    for i, (nome, ris) in enumerate(a.backtest.items()):
        dd = metriche.drawdown(ris.rendimenti).resample("W-FRI").min().round(4)
        fig.add_scatter(x=dd.index, y=dd, mode="lines", name=nome,
                        line={"color": COLORI[i % len(COLORI)], "width": 2.2 if i == 0 else 1.3},
                        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1%}<extra>" + nome + "</extra>")
    fig.update_yaxes(tickformat=".0%", title="Perdita dal massimo")
    fig.update_layout(hovermode="x unified")
    return _layout(fig, "Quanto si è perso dal massimo, strategia per strategia", altezza=440)


def grafico_rischio_rendimento(a: Analisi) -> go.Figure:
    """Rendimento annuo contro massima perdita: in alto a destra le strategie migliori."""
    m = a.metriche
    fig = go.Figure()
    for i, nome in enumerate(m.index):
        fig.add_scatter(x=[m.loc[nome, "Massima perdita"]], y=[m.loc[nome, "Rendimento annuo"]], mode="markers+text",
                        name=nome, text=[nome], textposition="top center",
                        marker={"size": 16 if i == 0 else 12, "color": COLORI[i % len(COLORI)]},
                        hovertemplate=(f"{nome}<br>Rendimento %{{y:.1%}}<br>Massima perdita %{{x:.1%}}<br>"
                                       f"Sharpe {m.loc[nome, 'Sharpe']:.2f}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", title="Massima perdita dal picco")
    fig.update_yaxes(tickformat=".1%", title="Rendimento annuo composto")
    fig.update_layout(showlegend=False)
    return _layout(fig, "Rendimento e massima perdita delle strategie", altezza=440)


def grafico_correlazione(a: Analisi) -> go.Figure:
    """Correlazione mobile a un anno tra azioni e obbligazioni governative."""
    c = a.correlazione.resample("W-FRI").last().round(3).rename(a.correlazione.name)
    fig = go.Figure()
    fig.add_scatter(x=c.index, y=c, mode="lines", line={"color": BLU_SCURO, "width": 2}, name=c.name,
                    fill="tozeroy", fillcolor="rgba(28,92,171,0.10)", hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f}")
    fig.add_hline(y=0, line={"color": GRIGIO, "width": 1})
    fig.update_yaxes(range=[-1, 1], title="Correlazione su 12 mesi")
    fig.update_layout(showlegend=False)
    return _layout(fig, "Azioni e obbligazioni: la diversificazione non è garantita", altezza=380)


GRAFICI = {
    "distribuzione": grafico_distribuzione,
    "metodi": grafico_metodi,
    "convalida": grafico_convalida,
    "stress": grafico_stress,
    "contributi": grafico_contributi,
    "correlazione": grafico_correlazione,
    "strategie": grafico_strategie,
    "drawdown": grafico_drawdown,
    "rischio_rendimento": grafico_rischio_rendimento,
}


def _disponibili(a: Analisi) -> dict:
    g = dict(GRAFICI)
    if a.correlazione is None:
        g.pop("correlazione")
    if not a.backtest:
        for k in ("strategie", "drawdown", "rischio_rendimento"):
            g.pop(k)
    return g


def salva_grafici(a: Analisi, cartella: str | Path, scala: float = 2.0) -> list[Path]:
    """Esporta i grafici in PNG (richiede ``kaleido`` e un browser Chrome/Chromium)."""
    out = Path(cartella)
    out.mkdir(parents=True, exist_ok=True)
    percorsi = []
    for nome, funzione in _disponibili(a).items():
        percorso = out / f"{nome}.png"
        funzione(a).write_image(percorso, width=900, scale=scala)
        percorsi.append(percorso)
    return percorsi


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x: float, decimali: int = 1) -> str:
    if pd.isna(x):
        return "–"
    return f"{x:.{decimali}%}".replace(".", ",").replace("-", "−")


def _num(x: float, decimali: int = 2) -> str:
    if pd.isna(x):
        return "–"
    return f"{x:.{decimali}f}".replace(".", ",").replace("-", "−")


def _eur(x: float) -> str:
    if pd.isna(x):
        return "–"
    testo = f"{abs(x):,.0f}".replace(",", ".")
    return f"−€ {testo}" if x < -0.5 else f"€ {testo}"


def _orizzonte(giorni: int) -> str:
    return {1: "un giorno", 5: "una settimana", 10: "dieci giorni", 21: "un mese", 63: "tre mesi"}.get(
        giorni, f"{giorni} giorni")


def _giudizio(a: Analisi) -> tuple[str, str]:
    """Una frase di sintesi e una classe CSS per il riquadro di esito."""
    c = a.convalida
    minimo_p = c[["p-value Kupiec", "p-value indipendenza"]].astype(float).min(axis=1)
    superati = minimo_p[minimo_p > 0.05].sort_values().index.tolist()
    peggiore = min(a.stress, key=lambda e: e.rendimento)
    var_mese = a.var_orizzonte.loc["Storico", "VaR"] * a.importo
    attuale, storico = a.var.loc["GARCH(1,1)", "VaR"], a.var.loc["Storico", "VaR"]
    clima = ("più tranquille" if attuale < 0.85 * storico else "più agitate" if attuale > 1.15 * storico
             else "in linea")
    testo = (f"In un mese normale il portafoglio non dovrebbe perdere più di {_eur(var_mese)} in "
             f"{round(a.livello * 100)} casi su 100. Lo scenario più severo, «{peggiore.nome}», costerebbe "
             f"{_eur(-peggiore.rendimento * a.importo)} ({_pct(peggiore.rendimento)}). Oggi le condizioni di mercato "
             f"sono {clima} rispetto alla media storica.")
    if superati:
        testo += f" Il modello più affidabile nel backtest del VaR è {superati[-1]}."
    esito = "ok" if -peggiore.rendimento < 0.15 else "attenzione" if -peggiore.rendimento < 0.30 else "ko"
    return testo, esito


def build_report(
    a: Analisi,
    output: str | Path,
    client_name: str = "Cliente",
    data_note: str = "",
    offline: bool = False,
) -> Path:
    """Scrive il report HTML. Con ``offline=True`` include plotly.js nel file (circa 4 MB)."""
    sintesi, esito = _giudizio(a)
    config = {"displaylogo": False, "responsive": True}
    grafici = {nome: f(a).to_html(full_html=False, include_plotlyjs=False, config=config)
               for nome, f in _disponibili(a).items()}
    lv = f"{a.livello:.0%}"
    vg = a.var.loc["GARCH(1,1)"]
    kpi = [
        (f"VaR {lv} a un giorno", _eur(a.var.loc["Storico", "VaR"] * a.importo)),
        (f"VaR {lv} a {_orizzonte(a.orizzonte)}", _eur(a.var_orizzonte.loc["Storico", "VaR"] * a.importo)),
        (f"CVaR {lv} a {_orizzonte(a.orizzonte)}", _eur(a.var_orizzonte.loc["Storico", "CVaR"] * a.importo)),
        ("Volatilità annua", _pct(a.var.loc["Storico", "Volatilità annua"])),
        ("Volatilità prevista oggi (GARCH)", _pct(vg["Volatilità annua"])),
        ("Massima perdita dal picco", _pct(metriche.drawdown(a.rendimenti).min())),
    ]
    metodi = [{"nome": m, "var1": _eur(a.var.loc[m, "VaR"] * a.importo), "varh": _eur(r["VaR"] * a.importo),
               "cvarh": _eur(r["CVaR"] * a.importo), "pct": _pct(r["VaR"], 2), "vol": _pct(r["Volatilità annua"])}
              for m, r in a.var_orizzonte.iterrows()]
    conv = [{"nome": m, "attese": _num(r["Violazioni attese"], 0), "effettive": int(r["Violazioni"]),
             "freq": _pct(r["Frequenza"], 2), "kupiec": _num(r["p-value Kupiec"], 3),
             "indip": _num(r["p-value indipendenza"], 3), "ultimi": int(r["Ultimi 250 giorni"]),
             "semaforo": r["Semaforo"], "ok": r["p-value Kupiec"] > 0.05 and r["p-value indipendenza"] > 0.05}
            for m, r in a.convalida.iterrows()]
    stress = [{"nome": e.nome, "tipo": e.tipo, "rend": _pct(e.rendimento), "impatto": _eur(e.rendimento * a.importo),
               "dd": _pct(e.perdita_massima), "descr": e.descrizione,
               "peggiore": e.contributi.idxmin(), "peggiore_val": _eur(e.contributi.min() * a.importo)}
              for e in a.stress]
    peggiori = [{"da": f"{r['Inizio']:%d/%m/%Y}", "a": f"{r['Fine']:%d/%m/%Y}", "rend": _pct(r["Rendimento"]),
                 "impatto": _eur(r["Rendimento"] * a.importo)} for _, r in a.peggiori_mesi.iterrows()]
    contributi = [{"nome": n, "peso": _pct(r["Peso"]), "vol": _pct(r["Quota della volatilità"]),
                   "cvar": _pct(r["Quota del CVaR"]), "eur": _eur(r["Contributo al CVaR"] * a.importo),
                   "forte": r["Quota del CVaR"] > r["Peso"] * 1.3}
                  for n, r in a.contributi.sort_values("Peso", ascending=False).iterrows() if r["Peso"] > 0]
    m = a.metriche
    colonne = ["Rendimento annuo", "Volatilità", "Sharpe", "Sortino", "Massima perdita", "Calmar",
               "Mesi sotto il massimo", "Mese peggiore", "Probabilità Sharpe > 0", "Rotazione annua"]
    formati = {"Sharpe": _num, "Sortino": _num, "Calmar": _num,
               "Mesi sotto il massimo": lambda v: _num(v, 0), "Rotazione annua": lambda v: _pct(v, 0)}
    strategie_righe = [{"nome": n, "valori": [formati.get(c, _pct)(m.loc[n, c]) for c in colonne]} for n in m.index]
    confronto = [{"nome": n, "alfa": _pct(r["Alfa annuo"], 2), "p": _num(r["p-value alfa"], 2),
                  "beta": _num(r["Beta"]), "r2": _num(r["R²"]), "te": _pct(r["Tracking error"])}
                 for n, r in a.confronto.iterrows()]
    annuali = [{"anno": anno, "valori": [_pct(v) for v in riga], "min": int(np.argmin(riga)),
                "max": int(np.argmax(riga))} for anno, riga in a.annuali.iterrows()]
    migliore_sharpe = m["Sharpe"].idxmax() if len(m) else ""
    migliore_dd = m["Massima perdita"].idxmax() if len(m) else ""

    env = Environment(loader=PackageLoader("risk_backtest", "templates"), autoescape=select_autoescape())
    html = env.get_template("report.html.j2").render(
        client=client_name,
        today=date.today().strftime("%d/%m/%Y"),
        periodo=f"{a.prezzi.index[0]:%d/%m/%Y} – {a.prezzi.index[-1]:%d/%m/%Y}",
        importo=_eur(a.importo),
        livello=lv,
        livello_casi=round(a.livello * 100),
        orizzonte=_orizzonte(a.orizzonte),
        sintesi=sintesi,
        esito=esito,
        kpi=kpi,
        pesi=[{"nome": n, "peso": _pct(w), "eur": _eur(w * a.importo)} for n, w in a.pesi.items() if w > 0],
        metodi=metodi,
        gradi_liberta=_num(a.parametri_t["gradi_liberta"], 2),
        garch={"alpha": _num(a.garch.alpha, 3), "beta": _num(a.garch.beta, 3),
               "persistenza": _num(a.garch.persistenza, 3), "lungo": _pct(a.garch.volatilita_lungo_periodo)},
        asimmetria=_num(stats.skew(a.rendimenti), 2),
        curtosi=_num(stats.kurtosis(a.rendimenti), 1),
        convalida=conv,
        giorni_convalida=int(a.convalida["Giorni"].iloc[0]),
        stress=stress,
        peggiori=peggiori,
        contributi=contributi,
        correlazione=a.correlazione is not None,
        strategie=strategie_righe,
        colonne=colonne,
        confronto=confronto,
        nomi_strategie=list(a.annuali.columns),
        annuali=annuali,
        migliore_sharpe=migliore_sharpe,
        migliore_dd=migliore_dd,
        inizio_backtest=f"{next(iter(a.backtest.values())).rendimenti.index[0]:%d/%m/%Y}" if a.backtest else "",
        grafici=grafici,
        costi=_num(a.costi_bps, 0),
        risk_free=_pct(a.risk_free),
        data_note=data_note,
        plotly_cdn=PLOTLY_CDN,
        plotly_inline=get_plotlyjs() if offline else None,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
