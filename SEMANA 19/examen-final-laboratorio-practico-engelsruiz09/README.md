[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/reHpHMQo)
# Ejercicio Comparación de Modelos de Clasificación

## Estructura

```
data/
    train.csv
metrics/
    evaluation_report.csv  # Debe ser generado por src/main.py
src/
    main.py             # Debe ser completado 
tests/
    *.py                # Tests automáticos
requirements.txt
```

## Calificacion Automática (70 pts)

| Criterio                                                      | Puntos |
|---------------------------------------------------------------|--------|
| main.py ejecuta sin errores y genera el archivo requerido     | 10     |
| El preprocesamiento trata los valores faltantes correctamente | 10     |
| Codificación de variables categóricas está presente           | 10     |
| Entrena y evalúa correctamente el modelo SVM                  | 10     |
| Entrena y evalúa correctamente el Perceptrón                  | 10     |
| Entrena y evalúa correctamente una red neuronal en Keras      | 10     |
| El archivo evaluation_report.csv contiene todas las métricas  | 10     |

## Ejecución

```bash
pip install -r requirements.txt
python src/main.py
pytest tests
```
