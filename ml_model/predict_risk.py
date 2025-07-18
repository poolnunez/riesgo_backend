import pandas as pd
import numpy as np
import joblib
import os

# --- PARÁMETROS DE RUTAS ---
model_load_path = 'ml_model/risk_model.joblib'

# --- Funciones de preprocesamiento (DEBEN SER IDÉNTICAS A LAS USADAS EN EL ENTRENAMIENTO) ---

def parse_age_range(age_str, mean_age=20.0): # Usar un valor por defecto si no se puede inferir la media del entrenamiento
    if isinstance(age_str, str) and '-' in age_str:
        parts = age_str.split('-')
        try:
            return (int(parts[0]) + int(parts[1])) / 2
        except ValueError:
            return mean_age # Valor por defecto si falla el parseo
    return mean_age # Para 'Other' o valores no esperados

def map_gender(gender_str):
    gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
    return gender_mapping.get(gender_str, -1) # -1 para valores no mapeados

def map_academic_year(year_str):
    academic_year_mapping = {
        'First Year or Equivalent': 1,
        'Second Year or Equivalent': 2,
        'Third Year or Equivalent': 3,
        'Fourth Year or Equivalent': 4,
        'Other': 0
    }
    return academic_year_mapping.get(year_str, 0) # 0 por defecto

def parse_cgpa_range(cgpa_str, mean_cgpa=3.195): # Usar un valor por defecto
    if isinstance(cgpa_str, str) and '-' in cgpa_str:
        parts = cgpa_str.split(' - ')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return mean_cgpa
    elif isinstance(cgpa_str, str) and cgpa_str == 'Below 2.50':
        return 2.0
    return mean_cgpa

# --- Función para cargar el modelo ---
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo del modelo no se encontró en: {path}")
    return joblib.load(path)

# --- Función para preprocesar nuevos datos (para una sola instancia) ---
def preprocess_new_data(data_point):
    """
    Preprocesa un diccionario de datos de entrada para que coincida con las características del modelo.
    data_point es un diccionario con claves como:
    'Depression Value', 'Anxiety Value', 'Stress Value', 'Age', 'Gender', 'Academic_Year', 'Current_CGPA'
    """
    processed_data = {
        'phq9_score': pd.to_numeric(data_point.get('Depression Value', 0), errors='coerce'),
        'gad7_score': pd.to_numeric(data_point.get('Anxiety Value', 0), errors='coerce'),
        'pss_score': pd.to_numeric(data_point.get('Stress Value', 0), errors='coerce'),
        'age_numeric': parse_age_range(data_point.get('Age', '20-20')), # Usar un rango por defecto que de la media
        'gender_numeric': map_gender(data_point.get('Gender', 'Other')),
        'academic_year_numeric': map_academic_year(data_point.get('Academic_Year', 'Other')),
        'current_cgpa_numeric': parse_cgpa_range(data_point.get('Current_CGPA', '3.00 - 3.39')) # Usar un rango por defecto que de la media
    }
    
    # Convertir a DataFrame en el orden esperado por el modelo
    # Asegúrate de que las columnas estén en el mismo orden que las features de entrenamiento
    features_order = [
        'phq9_score', 'gad7_score', 'pss_score',
        'age_numeric', 'gender_numeric', 'academic_year_numeric', 'current_cgpa_numeric'
    ]
    
    # Crear un DataFrame de una sola fila para la predicción
    df_single_instance = pd.DataFrame([processed_data])[features_order]
    
    # Rellenar cualquier NaN que pudiera surgir (aunque las funciones de parseo ya deberían manejarlos)
    df_single_instance = df_single_instance.fillna(df_single_instance.mean())
    
    return df_single_instance

# --- Ejemplo de uso ---
if __name__ == "__main__":
    try:
        # Cargar el modelo
        model = load_model(model_load_path)
        print(f"Modelo cargado exitosamente desde: {model_load_path}")

        # Ejemplo de nuevos datos (esto es lo que recibirías de una API o formulario)
        # Los nombres de las claves deben coincidir con los encabezados del CSV original
        new_data_point_1 = {
            'Age': '18-22',
            'Gender': 'Female',
            'University': 'Any University', # Estas columnas no se usan en el modelo pero podrian venir en la entrada
            'Department': 'Any Department',
            'Academic_Year': 'Second Year or Equivalent',
            'Current_CGPA': '2.50 - 2.99',
            'waiver_or_scholarship': 'No',
            'PSS1': 3, 'PSS2': 4, 'PSS3': 3, 'PSS4': 2, 'PSS5': 2, 'PSS6': 1, 'PSS7': 2, 'PSS8': 2, 'PSS9': 4, 'PSS10': 4,
            'Stress Value': 29, # Puntaje total PSS
            'Stress Label': 'High Perceived Stress',
            'GAD1': 2, 'GAD2': 2, 'GAD3': 3, 'GAD4': 2, 'GAD5': 2, 'GAD6': 2, 'GAD7': 2,
            'Anxiety Value': 15, # Puntaje total GAD-7
            'Anxiety Label': 'Severe Anxiety',
            'PHQ1': 2, 'PHQ2': 2, 'PHQ3': 3, 'PHQ4': 2, 'PHQ5': 2, 'PHQ6': 2, 'PHQ7': 2, 'PHQ8': 3, 'PHQ9': 2,
            'Depression Value': 20, # Puntaje total PHQ-9
            'Depression Label': 'Severe Depression'
        }

        new_data_point_2 = {
            'Age': '23-26',
            'Gender': 'Male',
            'Academic_Year': 'Fourth Year or Equivalent',
            'Current_CGPA': '3.80 - 4.00',
            'Depression Value': 5,  # Minimal depression
            'Anxiety Value': 2,     # Minimal anxiety
            'Stress Value': 10      # Low stress
            # Otros campos no necesarios para la predicción, pero podrían estar en los datos de entrada
        }
        
        # Preprocesar los nuevos datos
        processed_input_1 = preprocess_new_data(new_data_point_1)
        processed_input_2 = preprocess_new_data(new_data_point_2)

        # Realizar predicción
        prediction_1 = model.predict(processed_input_1)
        prediction_proba_1 = model.predict_proba(processed_input_1)
        
        prediction_2 = model.predict(processed_input_2)
        prediction_proba_2 = model.predict_proba(processed_input_2)

        print(f"\nPredicción para el Caso 1 (phq9=20, gad7=15, pss=29): Riesgo = {prediction_1[0]}")
        print(f"Probabilidades de riesgo (0, 1, 2): {prediction_proba_1[0]}")

        print(f"\nPredicción para el Caso 2 (phq9=5, gad7=2, pss=10): Riesgo = {prediction_2[0]}")
        print(f"Probabilidades de riesgo (0, 1, 2): {prediction_proba_2[0]}")

        # Interpretación del riesgo
        risk_labels = {0: "Riesgo Bajo/Mínimo", 1: "Riesgo Moderado", 2: "Riesgo Alto/Severo"}
        print(f"\nInterpretación Caso 1: {risk_labels.get(prediction_1[0], 'Desconocido')}")
        print(f"Interpretación Caso 2: {risk_labels.get(prediction_2[0], 'Desconocido')}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Ocurrió un error durante la predicción: {e}")