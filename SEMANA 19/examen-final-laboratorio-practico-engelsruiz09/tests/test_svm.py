import pandas as pd

def test_svm_metrics_exist():
    df = pd.read_csv("metrics/evaluation_report.csv")
    svm_rows = df[df['model'] == 'svm']
    assert not svm_rows.empty, "No se encontraron métricas para SVM"
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        assert metric in svm_rows.columns, f"Métrica {metric} faltante para SVM"
