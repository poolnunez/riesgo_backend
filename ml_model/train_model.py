import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# --- PARÁMETROS DE RUTAS ---
file_path_processed_data = 'ml_model/Processed.csv' # Nuevo path para el dataset combinado
model_save_path = 'ml_model/risk_model.joblib'

# --- PARTE 1: Procesar Processed.csv ---
df_processed = pd.DataFrame() # Inicializar para asegurar que exista

try:
    df_processed = pd.read_csv(file_path_processed_data)
    print(f"'{file_path_processed_data}' cargado exitosamente. Filas: {len(df_processed)}")

    # --- Preprocesamiento de columnas ---

    # 1. Procesar 'Age' (rangos a punto medio numérico)
    def parse_age_range(age_str):
        if isinstance(age_str, str) and '-' in age_str:
            parts = age_str.split('-')
            try:
                return (int(parts[0]) + int(parts[1])) / 2
            except ValueError:
                return np.nan
        return np.nan # Para 'Other' o valores no esperados
    df_processed['age_numeric'] = df_processed['Age'].apply(parse_age_range)
    df_processed['age_numeric'].fillna(df_processed['age_numeric'].mean(), inplace=True)
    print("Procesando columna 'Age'.")

    # 2. Procesar 'Gender' (categórico a numérico)
    gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2} # Añadir 'Other' si aparece
    df_processed['gender_numeric'] = df_processed['Gender'].map(gender_mapping).fillna(-1) # -1 para valores no mapeados
    print("Procesando columna 'Gender'.")

    # 3. Procesar 'Academic_Year' (categórico a numérico)
    academic_year_mapping = {
        'First Year or Equivalent': 1,
        'Second Year or Equivalent': 2,
        'Third Year or Equivalent': 3,
        'Fourth Year or Equivalent': 4,
        'Other': 0 # Mapear 'Other' a 0, o considerar NaN y llenar con la moda/media
    }
    df_processed['academic_year_numeric'] = df_processed['Academic_Year'].map(academic_year_mapping).fillna(0) # Default a 0
    print("Procesando columna 'Academic_Year'.")

    # 4. Procesar 'Current_CGPA' (rangos a punto medio numérico)
    def parse_cgpa_range(cgpa_str):
        if isinstance(cgpa_str, str) and '-' in cgpa_str:
            parts = cgpa_str.split(' - ')
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                return np.nan
        elif isinstance(cgpa_str, str) and cgpa_str == 'Below 2.50':
            return 2.0 # Punto medio representativo para "Below 2.50"
        return np.nan
    df_processed['current_cgpa_numeric'] = df_processed['Current_CGPA'].apply(parse_cgpa_range)
    df_processed['current_cgpa_numeric'].fillna(df_processed['current_cgpa_numeric'].mean(), inplace=True) # Rellenar NaNs con la media
    print("Procesando columna 'Current_CGPA'.")

    # 5. Obtener los puntajes directos de las escalas
    df_processed['phq9_score'] = pd.to_numeric(df_processed['Depression Value'], errors='coerce').fillna(0)
    df_processed['gad7_score'] = pd.to_numeric(df_processed['Anxiety Value'], errors='coerce').fillna(0)
    df_processed['pss_score'] = pd.to_numeric(df_processed['Stress Value'], errors='coerce').fillna(0) # PSS-10 score

    # 6. Derivar 'risk_level' combinado de Depression Label y Anxiety Label
    # Mapeo de etiquetas a niveles de riesgo numéricos (0: Bajo/Ninguno, 1: Moderado, 2: Alto/Severo)
    depression_risk_mapping = {
        'No Depression': 0, 'Minimal Depression': 0,
        'Mild Depression': 1, 'Moderate Depression': 1,
        'Moderately Severe Depression': 2, 'Severe Depression': 2
    }
    anxiety_risk_mapping = {
        'Minimal Anxiety': 0, 'Mild Anxiety': 1,
        'Moderate Anxiety': 1, 'Severe Anxiety': 2
    }

    df_processed['depression_risk'] = df_processed['Depression Label'].map(depression_risk_mapping).fillna(0).astype(int)
    df_processed['anxiety_risk'] = df_processed['Anxiety Label'].map(anxiety_risk_mapping).fillna(0).astype(int)

    # El nivel de riesgo general es el máximo entre el riesgo de depresión y ansiedad
    df_processed['risk_level'] = df_processed[['depression_risk', 'anxiety_risk']].max(axis=1)
    print("Derivando 'risk_level' combinado de 'Depression Label' y 'Anxiety Label'.")

    # 7. Añadir columna para Imposter Syndrome (se mantiene como 0 ya que no hay datos en este dataset)
    df_processed['imposter_syndrome_score'] = 0
    print("Columna 'imposter_syndrome_score' añadida (valores en 0).")

    # --- Seleccionar características finales para el modelo ---
    final_features_df = df_processed[[
        'phq9_score', 'gad7_score', 'pss_score',
        'age_numeric', 'gender_numeric', 'academic_year_numeric', 'current_cgpa_numeric',
        'risk_level'
    ]]

    print("\nPrimeras 5 filas del dataset procesado para entrenamiento:")
    print(final_features_df.head())
    print(f"\nDimensiones del dataset final procesado: {final_features_df.shape}")

except FileNotFoundError:
    print(f"Error: El archivo '{file_path_processed_data}' no se encontró. No se podrá entrenar el modelo.")
except Exception as e:
    print(f"Ocurrió un error inesperado al procesar el archivo Processed.csv: {e}")


# --- PARTE 2: Preparar para el Entrenamiento ---

if not df_processed.empty:
    
    # Definir características (X) y variable objetivo (y)
    features = [
        'phq9_score', 'gad7_score', 'pss_score',
        'age_numeric', 'gender_numeric', 'academic_year_numeric', 'current_cgpa_numeric'
    ]
    X = final_features_df[features]
    y = final_features_df['risk_level']

    # Asegurarse de que no haya NaN en las características (rellenar por si acaso, aunque ya hicimos fillna)
    X = X.fillna(X.mean())

    # --- PARTE 3: Entrenar y Guardar el Modelo ---
    print("\nEntrenando el modelo RandomForestClassifier...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    joblib.dump(model, model_save_path)
    print(f"Modelo entrenado y guardado exitosamente como: {model_save_path}")

else:
    print("\nNo se pudo procesar el dataset con éxito, el modelo no será entrenado.")