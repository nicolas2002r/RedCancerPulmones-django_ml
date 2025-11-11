# Desarrollo de un modelo de Machine Learning para predecir cáncer de pulmón el cualquier paciente
# Usando Redes Neuronales y Regresión Logística

# # Librerias necesarias
# Libreria para operaciones matemáticas
import numpy as np
# Libreria para el manejo de los datos
import pandas as pd

# Configurar matplotlib antes de importar pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Librería para codificar variables categóricas
from sklearn import preprocessing
# Librería para entrenamiento del modelo y prueba
from sklearn.model_selection import train_test_split
# librerias para metricas
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
# Libreria para MLP
from sklearn.neural_network import MLPClassifier
# Libreria para el modelo
from sklearn.linear_model import LogisticRegression
# Libreria para la matriz de confusión
from sklearn.metrics import confusion_matrix
# Libreria para escalar
from sklearn.preprocessing import StandardScaler
# Libreria para graficar
import seaborn as sns
# Libreria para la búsqueda en cuadrícula
from sklearn.model_selection import GridSearchCV

from django.shortcuts import render
import json
import io
import base64


# Desarrollo y analisis del modelo


def main(request):
    # * Plantilla con contexto inicial vacío *
    context = {
        "test": [],
        "prediction": [],
        "correlation_heatmap": "",
        "log_metrics": {
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
        },
        "mlp_metrics": {
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "confusion_matrix": [],
        },
        "mlp_optimized_metrics": {
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "confusion_matrix": [],
        }
    }
    return render(request, 'index.html', context=context)

def prediccion(request):
    if request.method != 'POST':
        # Si no es POST, redirigir a main
        return main(request)
    
    try:
        ### Se cargan los datos
        # Usar Path para manejar rutas correctamente
        from pathlib import Path
        csv_path = Path(__file__).resolve().parent / 'data' / 'Cancer_Pulmon.csv'
        df_data = pd.read_csv(csv_path)
        
        # Limpiar espacios de columnas
        df_data.columns = df_data.columns.str.strip()

        df_data = df_data.rename(columns={
            'GENDER': 'GENERO',
            'AGE': 'EDAD',
            'SMOKING': 'FUMADOR',
            'YELLOW_FINGERS': 'DEDOS_AMARILLOS',
            'ANXIETY': 'ANSIEDAD',
            'PEER_PRESSURE': 'PRESION_SOCIAL',
            'CHRONIC DISEASE': 'ENFERMEDAD_CRÓNICA',
            'FATIGUE': 'FATIGA',
            'ALLERGY': 'ALERGIA',
            'WHEEZING': 'SIBILANCIAS',
            'ALCOHOL CONSUMING': 'CONSUMO_ALCOHOL',
            'COUGHING': 'TOS',
            'SHORTNESS OF BREATH': 'FALTA_AIRE',
            'SWALLOWING DIFFICULTY': 'DIFICULTAD_TRAGAR',
            'CHEST PAIN': 'DOLOR_PECHO',
            'LUNG_CANCER': 'CANCER_PULMON'
        })

        # Eliminar duplicados
        df_data = df_data.drop_duplicates()

        # Codificar variables categóricas para el modelo
        le = preprocessing.LabelEncoder()
        df_data['GENERO'] = le.fit_transform(df_data['GENERO'])
        df_data['CANCER_PULMON'] = le.fit_transform(df_data['CANCER_PULMON'])
        df_data['FUMADOR'] = le.fit_transform(df_data['FUMADOR'])
        df_data['DEDOS_AMARILLOS'] = le.fit_transform(df_data['DEDOS_AMARILLOS'])
        df_data['ANSIEDAD'] = le.fit_transform(df_data['ANSIEDAD'])
        df_data['PRESION_SOCIAL'] = le.fit_transform(df_data['PRESION_SOCIAL'])
        df_data['ENFERMEDAD_CRÓNICA'] = le.fit_transform(df_data['ENFERMEDAD_CRÓNICA'])
        df_data['FATIGA'] = le.fit_transform(df_data['FATIGA'])
        df_data['ALERGIA'] = le.fit_transform(df_data['ALERGIA'])
        df_data['SIBILANCIAS'] = le.fit_transform(df_data['SIBILANCIAS'])
        df_data['CONSUMO_ALCOHOL'] = le.fit_transform(df_data['CONSUMO_ALCOHOL'])
        df_data['TOS'] = le.fit_transform(df_data['TOS'])
        df_data['FALTA_AIRE'] = le.fit_transform(df_data['FALTA_AIRE'])
        df_data['DIFICULTAD_TRAGAR'] = le.fit_transform(df_data['DIFICULTAD_TRAGAR'])
        df_data['DOLOR_PECHO'] = le.fit_transform(df_data['DOLOR_PECHO'])
        
        # Eliminar columnas Genero, edad, fumador y falta_aire
        df_nueva = df_data.drop(columns=['GENERO', 'EDAD', 'FUMADOR', 'FALTA_AIRE'], axis=1)

        # Describir el dataframe para entender sus metricas
        df_nueva.describe()

        # Mapa de calor para ver la correlación entre las variables
        cn = df_nueva.corr()
        cmap = sns.diverging_palette(260, -10, s=50, l=75, n=6, as_cmap=True)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cn, cmap=cmap, annot=True, square=True)

        # Guardar la figura en un objeto BytesIO en memoria
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()  # Cierra la figura para liberar memoria
        buffer.seek(0)
        
        # Convertir la imagen a base64 para incrustarla en HTML
        image_png = buffer.getvalue()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')

        # Se definen las caracteristicas
        features = ['DEDOS_AMARILLOS', 'ANSIEDAD', 'PRESION_SOCIAL', 'FATIGA', 'ALERGIA', 'SIBILANCIAS', 'CONSUMO_ALCOHOL', 'TOS', 'DIFICULTAD_TRAGAR', 'DOLOR_PECHO']
        X = df_nueva[features]

        # Variable objetivo
        y = df_nueva['CANCER_PULMON'].values

        # División de los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
        
        # Se selecciona el algoritmo
        modelo_predi = LogisticRegression()
        modelo_predi.fit(X_train, y_train)

        # Predicción generada
        predict = modelo_predi.predict(X_test)

        # Se convierten los valores a listas
        test = y_test.tolist()
        prediction = predict.tolist()

        # Se imprimen las métricas
        mo_regre_predi = round(precision_score(y_test, predict, average='weighted'), 2)
        mo_regre_recall = round(recall_score(y_test, predict, average='weighted'), 2) 
        mo_regre_f1score = round(f1_score(y_test, predict, average='weighted'), 2)

        # Modelo de Red Neuronal (MLP)
        mlp_model = MLPClassifier()
        #  Entrenar el modelo
        mlp_model.fit(X_train, y_train)

        # Predicciones generadas 
        mlp_predictions = mlp_model.predict(X_test)

        # Métricas del modelo MLP 
        mlp_confusion = confusion_matrix(y_test, mlp_predictions)
        mlp_precision = round(precision_score(y_test, mlp_predictions, average='weighted'), 2)
        mlp_recall = round(recall_score(y_test, mlp_predictions, average='weighted'), 2)
        mlp_f1 = round(f1_score(y_test, mlp_predictions, average='weighted'), 2)

        # Escalado de características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Definir el modelo base
        mlp = MLPClassifier()

        # Definir la cuadrícula de hiperparámetros a buscar
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        }

        # Grid Search con validación cruzada
        grid_search = GridSearchCV(estimator=mlp,
                                param_grid=param_grid,
                                cv=5,
                                scoring='f1_weighted',
                                n_jobs=-1,
                                verbose=1)

        # Entrenar el modelo con la búsqueda en cuadrícula
        grid_search.fit(X_train_scaled, y_train)

        # Mejor modelo encontrado
        best_mlp = grid_search.best_estimator_

        # Predicciones generadas con el modelo optimizado
        mlp_predictions_optimized = best_mlp.predict(X_test_scaled)

        opt_confusion = confusion_matrix(y_test, mlp_predictions_optimized)
        opt_precision = round(precision_score(y_test, mlp_predictions_optimized, average='weighted'), 2)
        opt_recall = round(recall_score(y_test, mlp_predictions_optimized, average='weighted'), 2)
        opt_f1 = round(f1_score(y_test, mlp_predictions_optimized, average='weighted'), 2)

        # Se envian los datos al template
        context = {
            "test": test,
            "prediction": prediction,
            "correlation_heatmap": graphic,
            "log_metrics": {
                "precision": mo_regre_predi,
                "recall": mo_regre_recall,
                "f1_score": mo_regre_f1score,
            },
            "mlp_metrics": {
                "precision": mlp_precision,
                "recall": mlp_recall,
                "f1_score": mlp_f1,
                "confusion_matrix": mlp_confusion.tolist(),
            },
            "mlp_optimized_metrics": {
                "precision": opt_precision,
                "recall": opt_recall,
                "f1_score": opt_f1,
                "confusion_matrix": opt_confusion.tolist(),
            }
        }

    except Exception as e:
        # Manejo de errores con mensaje más informativo
        print(f"Error en prediccion: {str(e)}")
        import traceback
        traceback.print_exc()
        
        context = {
            "test": [],
            "prediction": [],
            "correlation_heatmap": "",
            "error": str(e),  # Agregar mensaje de error
            "log_metrics": {
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
            },
            "mlp_metrics": {
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "confusion_matrix": [],
            },
            "mlp_optimized_metrics": {
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "confusion_matrix": [],
            }
        }

    # * Plantilla *
    return render(request, 'index.html', context=context)