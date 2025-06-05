import os
import pandas as pd

def test_metrics_file_exists_and_format():
    path = "metrics/evaluation_report.csv"
    assert os.path.exists(path), "evaluation_report.csv debe existir"
    df = pd.read_csv(path)
    expected = {'model','accuracy','precision','recall','f1_score'}
    assert expected.issubset(df.columns), "Columnas faltantes en CSV"
    assert len(df) == 3, "Deben ser tres modelos evaluados"
