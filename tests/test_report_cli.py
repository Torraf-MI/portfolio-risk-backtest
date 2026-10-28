import pytest

from risk_backtest import dati
from risk_backtest.cli import main
from risk_backtest.report import GRAFICI, analizza, build_report


@pytest.fixture(scope="module")
def analisi_demo(prezzi, pesi):
    return analizza(prezzi, pesi, importo=500_000)


def test_parse_pesi():
    colonne = ["Azionario Globale", "Obbligazioni Gov. EUR", "Oro"]
    p = dati.parse_pesi("Globale=60%, Gov=0.3, Oro=0.1", colonne)
    assert p.to_dict() == pytest.approx({"Azionario Globale": 0.6, "Obbligazioni Gov. EUR": 0.3, "Oro": 0.1})
    with pytest.raises(ValueError, match="sommare"):
        dati.parse_pesi("Oro=0.5", colonne)
    with pytest.raises(ValueError, match="ambiguo"):
        dati.parse_pesi("O=1", colonne)


def test_grafici(analisi_demo):
    for funzione in GRAFICI.values():
        assert funzione(analisi_demo).data


def test_report(analisi_demo, tmp_path):
    html = build_report(analisi_demo, tmp_path / "r.html", client_name="Test", offline=False).read_text()
    for testo in ["Analisi del rischio", "Backtest del VaR", "Stress test", "Crisi finanziaria 2008",
                  "Risk parity", "Semaforo di Basilea", "non costituisce consulenza"]:
        assert testo in html


def test_cli_demo(tmp_path, capsys):
    out = tmp_path / "report.html"
    assert main(["--demo", "--output", str(out), "--importo", "250000", "--senza-strategie"]) == 0
    assert out.exists()
    testo = capsys.readouterr().out
    assert "GARCH(1,1)" in testo and "Stress test" in testo


def test_cli_csv(tmp_path, prezzi):
    csv = tmp_path / "prezzi.csv"
    prezzi.loc["2012":, ["Azionario Globale", "Obbligazioni Gov. EUR", "Oro"]].to_csv(csv)
    out = tmp_path / "r.html"
    assert main(["--csv", str(csv), "--pesi", "Globale=0.6, Gov=0.3, Oro=0.1", "--livello", "0.95",
                 "--output", str(out)]) == 0
    assert "Rialzo dei tassi 2022" in out.read_text()
