"""Stress test: crisi storiche rivissute con il portafoglio di oggi e scenari ipotetici.

Negli scenari ipotetici si indica lo shock solo sui fattori principali (ad esempio le azioni); l'effetto
sugli altri asset è stimato con una regressione sui rendimenti mensili storici (``statsmodels``): è il
rendimento atteso di ciascun asset *condizionato* agli shock indicati.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .dati import EPISODI, Episodio, classe


@dataclass(frozen=True)
class Scenario:
    """Scenario ipotetico: shock istantaneo per classe di attività (azioni, governativi, corporate, oro…)."""

    nome: str
    shock: dict[str, float]
    descrizione: str = ""


SCENARI_IPOTETICI = [
    Scenario("Crollo azionario", {"azioni": -0.30},
             "Le borse perdono il 30%; gli altri asset reagiscono come hanno fatto in passato."),
    Scenario("Rialzo dei tassi di 2 punti", {"governativi": -0.12, "corporate": -0.11},
             "Rialzo improvviso dei rendimenti: con una duration di circa 6 anni le obbligazioni perdono il 12%."),
    Scenario("Stagflazione", {"azioni": -0.20, "governativi": -0.10, "corporate": -0.12, "oro": 0.15},
             "Inflazione alta e crescita debole: scendono insieme azioni e obbligazioni, sale l'oro."),
    Scenario("Crisi del credito", {"azioni": -0.25, "corporate": -0.12, "governativi": 0.05},
             "Allargamento degli spread: fuga verso i titoli di Stato più sicuri."),
]


@dataclass
class EsitoStress:
    nome: str
    tipo: str  # "Storico" o "Ipotetico"
    rendimento: float  # rendimento del portafoglio nello scenario
    perdita_massima: float  # massima perdita dal picco durante lo scenario (solo storici)
    rendimenti_asset: pd.Series = field(repr=False)
    contributi: pd.Series = field(repr=False)  # contributo di ogni asset al rendimento
    descrizione: str = ""
    giorni: int = 0


def stress_storico(prezzi: pd.DataFrame, pesi: pd.Series, episodio: Episodio) -> EsitoStress | None:
    """Rendimento del portafoglio di oggi se si ripetesse la crisi, senza ribilanciare durante la crisi."""
    finestra = prezzi.loc[episodio.inizio:episodio.fine, pesi.index]
    if len(finestra) < 2 or finestra.index[0] > pd.Timestamp(episodio.inizio) + pd.Timedelta(days=7):
        return None  # crisi non coperta dai dati
    # Si parte dalla chiusura del giorno prima dell'inizio della finestra
    prima = prezzi.loc[:episodio.inizio, pesi.index].iloc[:-1]
    base = prima.iloc[-1] if len(prima) else finestra.iloc[0]
    relativi = finestra / base
    valore = relativi @ pesi
    rendimenti_asset = relativi.iloc[-1] - 1
    return EsitoStress(
        nome=episodio.nome,
        tipo="Storico",
        rendimento=float(valore.iloc[-1] - 1),
        perdita_massima=float((valore / np.maximum.accumulate(np.r_[1.0, valore.to_numpy()])[1:] - 1).min()),
        rendimenti_asset=rendimenti_asset,
        contributi=rendimenti_asset * pesi,
        descrizione=f"Dal {pd.Timestamp(episodio.inizio):%d/%m/%Y} al {pd.Timestamp(episodio.fine):%d/%m/%Y}",
        giorni=len(finestra),
    )


def peggiori_periodi(rp: pd.Series, giorni: int = 21, n: int = 3) -> pd.DataFrame:
    """I periodi non sovrapposti di ``giorni`` sedute con il rendimento peggiore del portafoglio."""
    cumulato = (1 + rp).cumprod()
    mobile = (cumulato / cumulato.shift(giorni) - 1).dropna().sort_values()
    scelti: list[pd.Timestamp] = []
    posizioni = {d: i for i, d in enumerate(rp.index)}
    for data in mobile.index:
        if all(abs(posizioni[data] - posizioni[s]) >= giorni for s in scelti):
            scelti.append(data)
        if len(scelti) == n:
            break
    righe = [{"Inizio": rp.index[posizioni[d] - giorni + 1], "Fine": d, "Rendimento": mobile[d]} for d in scelti]
    return pd.DataFrame(righe)


def sensibilita(rendimenti_mensili: pd.DataFrame, fattori: list[str], asset: str) -> pd.Series:
    """Coefficienti della regressione dei rendimenti mensili di ``asset`` sui ``fattori`` (OLS)."""
    y = rendimenti_mensili[asset]
    x = sm.add_constant(rendimenti_mensili[fattori])
    return sm.OLS(y, x).fit().params[fattori]


def stress_ipotetico(prezzi: pd.DataFrame, pesi: pd.Series, scenario: Scenario,
                     classi: dict[str, str | None] | None = None) -> EsitoStress | None:
    """Applica gli shock dello scenario; gli asset non colpiti direttamente seguono la regressione storica."""
    classi = classi or {a: classe(a) for a in pesi.index}
    colpiti = [a for a in pesi.index if classi.get(a) in scenario.shock]
    if not colpiti:
        return None
    mensili = prezzi[pesi.index].resample("ME").last().pct_change().dropna()
    shock = pd.Series({a: scenario.shock[classi[a]] for a in colpiti})
    # Per la regressione basta un rappresentante per classe: asset della stessa classe sono troppo collineari
    rappresentanti = {}
    for a in colpiti:
        rappresentanti.setdefault(classi[a], a)
    fattori = list(rappresentanti.values())
    shock_fattori = shock[fattori]
    rendimenti_asset = pd.Series(0.0, index=pesi.index)
    for a in pesi.index:
        if a in shock:
            rendimenti_asset[a] = shock[a]
        elif mensili[a].std() > 0:
            rendimenti_asset[a] = float(sensibilita(mensili, fattori, a) @ shock_fattori)
    return EsitoStress(
        nome=scenario.nome,
        tipo="Ipotetico",
        rendimento=float(rendimenti_asset @ pesi),
        perdita_massima=float("nan"),
        rendimenti_asset=rendimenti_asset,
        contributi=rendimenti_asset * pesi,
        descrizione=scenario.descrizione,
    )


def stress_test(prezzi: pd.DataFrame, pesi: pd.Series, episodi: list[Episodio] | None = None,
                scenari: list[Scenario] | None = None) -> list[EsitoStress]:
    """Tutti gli stress test applicabili ai dati: crisi storiche coperte e scenari ipotetici."""
    episodi = [e for e in (EPISODI if episodi is None else episodi) if e.stress]
    scenari = SCENARI_IPOTETICI if scenari is None else scenari
    esiti = [stress_storico(prezzi, pesi, e) for e in episodi]
    esiti += [stress_ipotetico(prezzi, pesi, s) for s in scenari]
    return [e for e in esiti if e is not None]


def tabella_stress(esiti: list[EsitoStress], importo: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Scenario": e.nome, "Tipo": e.tipo, "Rendimento": e.rendimento, "Impatto": e.rendimento * importo,
          "Perdita massima": e.perdita_massima, "Periodo": e.descrizione if e.tipo == "Storico" else ""}
         for e in esiti]
    ).set_index("Scenario")
