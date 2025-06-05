import pandas as pd
import numpy as np
import os
import sys
import sklearn
from pathlib import Path
import joblib
from collections import defaultdict


# Importaciones ML
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.regularizers import l2

# Control de semillas aleatorias
RANDOM_STATE = 1337
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

class HRDataPipeline:
    """Pipeline especializado para datos de recursos humanos"""
    
    def __init__(self):
        self.ohe = None
        self.numerical_scaler = MinMaxScaler()
        self.processed = False
        
    def detect_feature_types(self, dataset):
        """Detección automática de tipos de características"""
        categorical_features = [
            'city', 'gender', 'relevent_experience', 'enrolled_university',
            'education_level', 'major_discipline', 'experience',
            'company_size', 'company_type', 'last_new_job'
        ]
        
        numerical_features = [col for col in dataset.columns 
                            if col not in categorical_features + ['target', 'enrollee_id']]
        
        return categorical_features, numerical_features
    
    def clean_missing_values(self, data):
        """Estrategia avanzada de limpieza de datos faltantes"""
        cleaned_data = data.copy()
        
        # Identificar tipos de features
        cat_features, num_features = self.detect_feature_types(cleaned_data)
        
        # Limpieza categórica: usar estrategia de frecuencia
        for feature in cat_features:
            if feature in cleaned_data.columns:
                # Rellenar con el valor más frecuente o crear categoría especial
                most_common = cleaned_data[feature].mode()
                fill_value = most_common[0] if len(most_common) > 0 else 'sin_datos'
                cleaned_data[feature] = cleaned_data[feature].fillna(fill_value)
        
        # Limpieza numérica: usar interpolación inteligente
        for feature in num_features:
            if feature in cleaned_data.columns:
                # Usar percentil 50 (mediana) para robustez
                median_value = cleaned_data[feature].quantile(0.5)
                cleaned_data[feature] = cleaned_data[feature].fillna(median_value)
        
        return cleaned_data
    
    def transform_categorical_data(self, dataset: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
        """Transformación optimizada de datos categóricos"""
        cat_cols, _ = self.detect_feature_types(dataset)
        existing_cat_cols = [c for c in cat_cols if c in dataset.columns]

        # Codificar las columnas originales a enteros                 
        encoded_ints = {
            col: dataset[col].astype("category").cat.codes.astype("int16")
            for col in existing_cat_cols
        }
        int_df = pd.DataFrame(encoded_ints, index=dataset.index)

        # One-hot 
        if fit_encoders:
            ohe_kwargs = dict(handle_unknown="ignore", dtype=np.int8,
                            sparse_output=False) if sklearn.__version__ >= "1.4" else \
                            dict(handle_unknown="ignore", dtype=np.int8, sparse=False)
            self.ohe = OneHotEncoder(**ohe_kwargs)
            cat_encoded = self.ohe.fit_transform(dataset[existing_cat_cols].astype(str))
        else:
            cat_encoded = self.ohe.transform(dataset[existing_cat_cols].astype(str))

        cat_df = pd.DataFrame(cat_encoded,
                            columns=self.ohe.get_feature_names_out(existing_cat_cols),
                            index=dataset.index)

        numeric_df = dataset.drop(columns=existing_cat_cols)

        # Concatenar: numéricas + *categóricas codificadas en entero* + one-hot
        transformed = pd.concat([numeric_df, int_df, cat_df], axis=1) #asegura que solo las columnas codificadas (one-hot) se guarden en train.csv
        return transformed

    
    def execute_pipeline(self, train_path, test_path):
        """Ejecutar pipeline completo de procesamiento para ambos datasets"""
        print("Cargando datos de entrenamiento desde:", train_path)
        print("Cargando datos de prueba desde:", test_path)
        
        # Cargar ambos datasets
        train_dataset = pd.read_csv(train_path)
        test_dataset = pd.read_csv(test_path)
        
        print(f"Dataset de entrenamiento: {train_dataset.shape}")
        print(f"Dataset de prueba: {test_dataset.shape}")
        
        # Aplicar limpieza a ambos datasets
        clean_train = self.clean_missing_values(train_dataset)
        clean_test = self.clean_missing_values(test_dataset)
        
        # Aplicar transformaciones - primero ajustar encoders con datos de entrenamiento
        final_train = self.transform_categorical_data(clean_train, fit_encoders=True)
        # Luego aplicar encoders ya ajustados a datos de prueba
        final_test = self.transform_categorical_data(clean_test, fit_encoders=False)
        
        # Marcar como procesado
        self.processed = True
        
        return final_train, final_test

class MLModelFactory:
    """Factory para creación y entrenamiento de modelos ML"""
    
    def __init__(self):
        self.trained_models = {}
        self.model_configs = {}
    
    def build_svm_model(self, X_data, y_data):
        """Constructor de modelo SVM optimizado"""
        # Configuración personalizada para SVM
        svm_config = {
            'kernel': 'rbf', #kernel RBF para no-linealidad 
            'random_state': RANDOM_STATE,
            'class_weight': 'balanced'
        }
        
        # Crear y entrenar modelo
        svm_classifier = SVC(**svm_config)
        svm_classifier.fit(X_data, y_data)
        
        # Almacenar modelo y configuración
        self.trained_models['svm'] = svm_classifier
        self.model_configs['svm'] = svm_config
        
        return svm_classifier
    
    def build_perceptron_model(self, X_data, y_data):
        """Constructor de modelo Perceptrón optimizado"""
        # Configuración personalizada para Perceptrón
        perceptron_config = {
            'max_iter': 1000,
            'random_state': RANDOM_STATE,
            'class_weight': 'balanced'
        }
        
        # Crear y entrenar modelo
        perceptron_classifier = Perceptron(**perceptron_config)
        perceptron_classifier.fit(X_data, y_data)
        
        # Almacenar modelo y configuración
        self.trained_models['perceptron'] = perceptron_classifier
        self.model_configs['perceptron'] = perceptron_config
        
        return perceptron_classifier
    
    def build_neural_network_model(self, X_data, y_data, input_dim):
        """Constructor de red neuronal con arquitectura personalizada"""
        
        # Definir entrada
        input_layer = Input(shape=(input_dim,))
        
        # Capas ocultas con regularización
        hidden1 = layers.Dense(256, activation='relu', 
                              kernel_regularizer=l2(0.01))(input_layer)
        dropout1 = layers.Dropout(0.05)(hidden1)
        
        hidden2 = layers.Dense(128, activation='relu',
                              kernel_regularizer=l2(0.01))(dropout1)
        dropout2 = layers.Dropout(0.1)(hidden2)
        
        hidden3 = layers.Dense(64, activation='relu')(dropout2)
        
        # Capa de salida
        output_layer = layers.Dense(1, activation='sigmoid')(hidden3)
        
        # Crear modelo
        neural_model = Model(inputs=input_layer, outputs=output_layer)
        
        # Configurar optimizador
        optimizer = Adam()
        
        # Compilar
        neural_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenar con configuración personalizada
        training_history = neural_model.fit(
            X_data, y_data,
            epochs=80,
            batch_size=128,
            validation_split=0.2,
            verbose=0,
            shuffle=True
        )
        
        # Almacenar modelo
        self.trained_models['nn'] = neural_model
        
        return neural_model

class PerformanceAnalyzer:
    """Analizador avanzado de rendimiento de modelos"""
    
    def __init__(self):
        self.results_storage = {}
    
    def compute_metrics(self, true_labels, predicted_labels):
        """Cálculo detallado de métricas de rendimiento"""
        performance_metrics = {}
        
        # Calcular métricas básicas
        performance_metrics['accuracy'] = round(
            accuracy_score(true_labels, predicted_labels), 4
        )
        
        performance_metrics['precision'] = round(
            precision_score(true_labels, predicted_labels, zero_division=0), 4
        )
        
        performance_metrics['recall'] = round(
            recall_score(true_labels, predicted_labels, zero_division=0), 4
        )
        
        performance_metrics['f1'] = round(
            f1_score(true_labels, predicted_labels, zero_division=0), 4
        )
        
        # Agregar f1_score duplicado para compatibilidad con tests
        performance_metrics['f1_score'] = performance_metrics['f1']
        
        return performance_metrics
    
    def analyze_model_performance(self, model_name, true_labels, predictions):
        """Análisis completo de rendimiento de un modelo"""
        metrics = self.compute_metrics(true_labels, predictions)
        metrics['model'] = model_name
        
        # Almacenar resultados
        self.results_storage[model_name] = metrics
        
        return metrics
    
    def generate_performance_report(self):
        """Generar reporte consolidado de rendimiento"""
        if not self.results_storage:
            return pd.DataFrame()
        
        # Convertir a DataFrame
        results_list = list(self.results_storage.values())
        report_df = pd.DataFrame(results_list)
        
        # Reordenar columnas según especificaciones
        column_order = ['model', 'accuracy', 'precision', 'recall', 'f1', 'f1_score']
        report_df = report_df[column_order]
        
        return report_df

def setup_project_structure():
    """Configurar estructura de directorios del proyecto"""
    directories = ['processed', 'metrics']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directorio {directory}/ configurado correctamente")

def main():
    """Función principal del sistema de analisis HR"""
    
    print("Iniciando Sistema de Analisis HR Analytics")
    print("=" * 55)
    
    # Configurar estructura del proyecto
    setup_project_structure()
    
    # Inicializar componentes del sistema
    data_pipeline = HRDataPipeline()
    model_factory = MLModelFactory()
    performance_analyzer = PerformanceAnalyzer()
    
    # FASE 1: Procesamiento de Datos
    print("\nFASE 1: Procesamiento y Limpieza de Datos")
    
    # Procesar ambos datasets
    train_processed, test_processed = data_pipeline.execute_pipeline(
        "data/aug_train.csv", 
        "data/aug_test.csv"
    )
    
    # Guardar datos procesados (solo el de entrenamiento como train.csv)
    output_path = "processed/train.csv"
    
    # Crear directorio si no existe
    os.makedirs("processed", exist_ok=True)
    
    # Intentar guardar el archivo con manejo de errores
    try:
        train_processed.to_csv(output_path, index=False)
        print(f"Datos de entrenamiento procesados guardados en: {output_path}")
    except PermissionError:
        print(f"Error de permisos al guardar {output_path}. Intentando alternativa...")
        # Intentar con nombre alternativo
        alt_path = "processed/train_backup.csv"
        train_processed.to_csv(alt_path, index=False)
        print(f"Datos guardados en: {alt_path}")
        output_path = alt_path
    
    # FASE 2: Preparación para Entrenamiento
    print("\nFASE 2: Preparación para Entrenamiento")
    
    # Separar caracteristicas y target del conjunto de entrenamiento
    # Excluir las columnas categoricas originales para el entrenamiento
    categorical_features = [
        'city', 'gender', 'relevent_experience', 'enrolled_university',
        'education_level', 'major_discipline', 'experience',
        'company_size', 'company_type', 'last_new_job'
    ]
    
    columns_to_exclude = ['target', 'enrollee_id'] + categorical_features
    X_train = train_processed.drop(columns=[col for col in columns_to_exclude if col in train_processed.columns])
    y_train = train_processed['target'].values
    
    # Preparar conjunto de prueba (no tiene target, pero necesitamos simular evaluación)
    # Para fines de evaluación, usaremos una división 70-30 del conjunto de entrenamiento
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y_train
    )
    
    # Normalización de características
    X_train_normalized = data_pipeline.numerical_scaler.fit_transform(X_train_split)
    X_test_normalized = data_pipeline.numerical_scaler.transform(X_test_split)
    
    print(f"Datos de entrenamiento: {X_train_normalized.shape}")
    print(f"Datos de prueba: {X_test_normalized.shape}")
    
    # FASE 3: Entrenamiento de Modelos
    print("\nFASE 3: Entrenamiento de Modelos de Machine Learning")
    
    # Entrenar SVM
    print("   Entrenando modelo SVM...")
    svm_model = model_factory.build_svm_model(X_train_normalized, y_train_split)
    
    # Entrenar Perceptrón
    print("   Entrenando modelo Perceptrón...")
    perceptron_model = model_factory.build_perceptron_model(X_train_normalized, y_train_split)
    
    # Entrenar Red Neuronal
    print("   Entrenando Red Neuronal...")
    nn_model = model_factory.build_neural_network_model(
        X_train_normalized, y_train_split, X_train_normalized.shape[1]
    )
    
    # FASE 4: Evaluación y Predicciones
    print("\nFASE 4: Evaluación de Modelos")
    
    # Generar predicciones
    svm_predictions = svm_model.predict(X_test_normalized)
    perceptron_predictions = perceptron_model.predict(X_test_normalized)
    nn_raw_predictions = nn_model.predict(X_test_normalized)
    nn_predictions = (nn_raw_predictions >= 0.5).astype(int).flatten()
    
    # Analizar rendimiento de cada modelo
    svm_results = performance_analyzer.analyze_model_performance(
        'svm', y_test_split, svm_predictions
    )
    
    perceptron_results = performance_analyzer.analyze_model_performance(
        'perceptron', y_test_split, perceptron_predictions
    )
    
    nn_results = performance_analyzer.analyze_model_performance(
        'nn', y_test_split, nn_predictions
    )
    
    # FASE 5: Generación de Reportes
    print("\nFASE 5: Generación de Reportes Finales")
    
    # Crear reporte consolidado
    final_report = performance_analyzer.generate_performance_report()
    
    # Guardar reporte CSV
    report_path = "metrics/evaluation_report.csv"
    try:
        final_report.to_csv(report_path, index=False)
        print(f"Reporte de evaluación guardado en: {report_path}")
    except PermissionError:
        print(f"Error de permisos al guardar {report_path}")
        alt_report_path = "metrics/evaluation_report_backup.csv"
        final_report.to_csv(alt_report_path, index=False)
        print(f"Reporte guardado en: {alt_report_path}")
    
    # Mostrar resultados en consola
    print("\nRESULTADOS FINALES:")
    print("=" * 55)
    
    display_columns = ['model', 'accuracy', 'precision', 'recall', 'f1']
    results_display = final_report[display_columns]
    
    for _, row in results_display.iterrows():
        print(f"{row['model'].upper():>12} | "
              f"Acc: {row['accuracy']:.4f} | "
              f"Prec: {row['precision']:.4f} | "
              f"Recall: {row['recall']:.4f} | "
              f"F1: {row['f1']:.4f}")
    
    # Identificar mejor modelo
    best_model_row = final_report.loc[final_report['f1'].idxmax()]
    print(f"\n MEJOR MODELO: {best_model_row['model'].upper()} (F1-Score: {best_model_row['f1']:.4f})")
    
    print("\n Análisis HR Analytics completado exitosamente!")

if __name__ == "__main__":
    main()