import pandas as pd

def test_nn_metrics_exist():
    df = pd.read_csv("metrics/evaluation_report.csv")
    nn_rows = df[df['model'] == 'nn']
    assert not nn_rows.empty, "No se encontraron métricas para Red Neuronal"
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        assert metric in nn_rows.columns, f"Métrica {metric} faltante para Red Neuronal"
