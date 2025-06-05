import pandas as pd

def test_no_missing_values():
    df = pd.read_csv("processed/train.csv")  
    null_count = df.isnull().sum().sum()
    assert null_count == 0, f"El dataset aún contiene {null_count} valores faltantes"

def test_imputation_code_present():
    with open("main.py", encoding="utf-8") as f:
        code = f.read()

    found = any(keyword in code for keyword in [
        'fillna', 'SimpleImputer', 'KNNImputer', 'IterativeImputer'
    ])

    assert found, "No se detectó uso de técnicas de imputación en el código fuente"
