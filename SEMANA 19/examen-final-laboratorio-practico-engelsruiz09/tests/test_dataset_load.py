import pandas as pd
import os

def test_dataset_loads_correctly():
    filepath = 'data/train.csv' 
    assert os.path.exists(filepath), "El archivo 'dataset.csv' no se encuentra en la carpeta 'data/'."

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        assert False, f"No se pudo cargar el archivo CSV correctamente: {e}"

    # Validaciones básicas del dataset
    expected_columns = [
        'enrollee_id', 'city', 'city_development_index', 'gender',
        'relevent_experience', 'enrolled_university', 'education_level',
        'major_discipline', 'experience', 'company_size', 'company_type',
        'last_new_job', 'training_hours', 'target'
    ]

    missing_columns = [col for col in expected_columns if col not in df.columns]
    assert not missing_columns, f"Faltan columnas requeridas: {missing_columns}"

    assert len(df) > 0, "El dataset está vacío."

    assert df['target'].dropna().isin([0, 1]).all(), "La columna 'target' debe contener solo valores binarios (0 o 1)."
