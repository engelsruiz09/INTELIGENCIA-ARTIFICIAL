import pandas as pd

def test_encoding_applied_correctly():
    df = pd.read_csv("processed/train.csv") 

    categorical_columns = [
        'city',
        'gender',
        'relevent_experience',
        'enrolled_university',
        'education_level',
        'major_discipline',
        'experience',
        'company_size',
        'company_type',
        'last_new_job'
    ]

    for col in categorical_columns:
        assert col in df.columns, f"La columna '{col}' no está presente en el dataset procesado"
        assert df[col].dtype.kind in 'iu', f"La columna '{col}' no fue codificada correctamente (tipo: {df[col].dtype})"

def test_encoding_code_used():
    with open("main.py", encoding="utf-8") as f:
        code = f.read()

    found = any(term in code for term in [
        'LabelEncoder', 'get_dummies', 'OneHotEncoder',
        '.fit_transform(', '.transform(', '.fit(', '.map(', '.replace('
    ])

    assert found, "No se detectó uso de técnicas de codificación categórica en main.py"