import pandas as pd

def test_perceptron_metrics_exist():
    df = pd.read_csv("metrics/evaluation_report.csv")
    perceptron_rows = df[df['model'] == 'perceptron']
    assert not perceptron_rows.empty, "No se encontraron métricas para Perceptrón"
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        assert metric in perceptron_rows.columns, f"Métrica {metric} faltante para Perceptrón"
