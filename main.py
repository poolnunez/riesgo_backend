# main.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
import os
import pandas as pd
from datetime import datetime, timedelta

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship # Importar 'relationship' aquí
from passlib.context import CryptContext

from jose import JWTError, jwt

# --- Inicializar la aplicación FastAPI ---
app = FastAPI(
    title="API del Modelo Predictivo de Riesgo en Salud Mental Universitaria",
    description="API que predice el nivel de riesgo (Bajo/Mínimo, Moderado, Alto/Severo) "
                "basado en puntuaciones de PSS, GAD-7, PHQ-9 y datos demográficos/académicos. "
                "Incluye gestión de usuarios, autenticación JWT y diario de sueño.",
    version="1.0.0"
)

# --- Configuración CORS ---
origins = [
    "http://localhost",
    "http://localhost:5173", # Asegúrate de que esta sea la URL de tu frontend
    "https://riesgo-frontend.onrender.com",  # Frontend en producción
    # Agrega más orígenes si tu frontend se aloja en otro lugar
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURACIÓN DE LA BASE DE DATOS ---
DATABASE_URL = "sqlite:///./test.db" # Asegúrate de que esta URL sea la correcta
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- MODELOS DE LA BASE DE DATOS (SQLAlchemy) ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="student")
    
    # Relación con SleepEntry
    sleep_entries = relationship("SleepEntry", back_populates="user")

class SleepEntry(Base):
    __tablename__ = "sleep_entries"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    hours = Column(Float, nullable=False)
    quality = Column(String, nullable=False) # 'muy_mala', 'promedio', 'buena', etc.
    bedtime_routine = Column(String, nullable=True)
    feeling_upon_waking = Column(String, nullable=False) # 'cansado', 'fresco', etc.
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relación con el usuario
    user = relationship("User", back_populates="sleep_entries")

# Declarar la variable global 'model' para el modelo de ML
model = None # Se inicializará en el startup_event

# --- Carga del Modelo y Creación de Tablas al Inicio (Startup Event) ---
@app.on_event("startup")
async def startup_event():
    global model # <--- Importante para modificar la variable global 'model'

    # Conexión y creación de tablas de la base de datos
    Base.metadata.create_all(bind=engine)
    print("Base de datos y tablas creadas/verificadas.")

    # --- CARGA DEL MODELO DE ML ---
    model_path = os.path.join(os.path.dirname(__file__), 'ml_model', 'risk_model.joblib')
    try:
        model = joblib.load(model_path)
        print(f"Modelo de IA '{os.path.basename(model_path)}' cargado exitosamente.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo en {model_path}.")
        print("Asegúrate de que 'train_model.py' se haya ejecutado exitosamente y 'risk_model.joblib' exista.")
        model = None
    except Exception as e:
        print(f"Error al cargar el modelo de IA: {e}")
        model = None
    # --- FIN CARGA DEL MODELO ---


# --- Dependencia para obtener la sesión de la base de datos ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- SEGURIDAD DE CONTRASEÑAS Y JWT ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "tu_super_clave_secreta_aqui_cambiala_en_produccion_1234567890" # <-- CAMBIA ESTO POR UNA CLAVE SEGURA Y ÚNICA
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_email_or_username(db: Session, identifier: str):
    user = db.query(User).filter(User.email == identifier).first()
    if not user:
        user = db.query(User).filter(User.username == identifier).first()
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email_or_username: str = payload.get("sub")
        if email_or_username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user_by_email_or_username(db, identifier=email_or_username)
    if user is None:
        raise credentials_exception
    return user
# --- FIN SEGURIDAD DE CONTRASEÑAS Y JWT ---


# --- FUNCIONES DE PREPROCESAMIENTO (AHORA COINCIDEN CON TRAIN_MODEL.PY) ---
def map_gender(gender_str: str, default_value: int = -1) -> int:
    gender_mapping = {'Femenino': 0, 'Masculino': 1, 'Otro': 2}
    return gender_mapping.get(gender_str, default_value)

def map_academic_year(year_str: str, default_value: int = 0) -> int:
    academic_year_mapping = {
        'Primer Año o Equivalente': 1,
        'Segundo Año o Equivalente': 2,
        'Tercer Año o Equivalente': 3,
        'Cuarto Año o Equivalente': 4,
        'Quinto Año o Equivalente': 5,
        'Sexto Año o Equivalente': 6,
        'Séptimo Año o Equivalente': 7,
        'Octavo Año o Equivalente': 8,
        'Noveno Año o Equivalente': 9,
        'Décimo Año o Equivalente': 10,
        'Otro': 0
    }
    return academic_year_mapping.get(year_str, default_value)

def map_categorical_cgpa_to_numeric(cgpa_cat_str: str, default_value: float = 3.195) -> float:
    mapping = {
        "Less than 2.00": 1.5,
        "2.00 - 2.29": 2.15,
        "2.30 - 2.69": 2.5,
        "2.70 - 2.99": 2.85,
        "3.00 - 3.39": 3.2,
        "3.39 - 3.69": 3.54,
        "3.70 - 4.00": 3.85
    }
    if cgpa_cat_str == "Below 2.50":
        return 2.0
    return mapping.get(cgpa_cat_str, default_value)

def parse_age_range_for_main(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        parts = age_str.split('-')
        try:
            return (int(parts[0]) + int(parts[1])) / 2
        except ValueError:
            return 25.0
    try:
        return float(age_str)
    except (ValueError, TypeError):
        return 25.0

def preprocess_for_prediction(data_point: dict) -> pd.DataFrame:
    """
    Preprocesa un diccionario de datos de entrada para que coincida con las características del modelo.
    """
    initial_data = {
        'Depression_Score': data_point.get('Depression_Score', 0),
        'Anxiety_Score': data_point.get('Anxiety_Score', 0),
        'Stress_Score': data_point.get('Stress_Score', 0),
        'Age': data_point.get('Age', "20-25"),
        'Gender': data_point.get('Gender', 'Otro'),
        'Academic_Year': data_point.get('Academic_Year', 'Otro'),
        'Current_CGPA': data_point.get('Current_CGPA', '3.00 - 3.39'),
    }
    final_features = {
        'phq9_score': initial_data['Depression_Score'],
        'gad7_score': initial_data['Anxiety_Score'],
        'pss_score': initial_data['Stress_Score'],
        'age_numeric': parse_age_range_for_main(initial_data['Age']),
        'gender_numeric': map_gender(initial_data['Gender']),
        'academic_year_numeric': map_academic_year(initial_data['Academic_Year']),
        'current_cgpa_numeric': map_categorical_cgpa_to_numeric(initial_data['Current_CGPA']),
    }
    
    features_order = [
        'phq9_score', 'gad7_score', 'pss_score',
        'age_numeric', 'gender_numeric', 'academic_year_numeric', 'current_cgpa_numeric'
    ]
    
    df_single_instance = pd.DataFrame([final_features])[features_order]
    df_single_instance = df_single_instance.fillna(0)
    
    return df_single_instance

# --- MODELOS Pydantic ---
class PredictRequest(BaseModel):
    Depression_Score: int = Field(..., ge=0, description="Puntuación total de depresión (suma del PHQ-9).")
    Anxiety_Score: int = Field(..., ge=0, description="Puntuación total de ansiedad (suma del GAD-7).")
    Stress_Score: int = Field(..., ge=0, description="Puntuación total de estrés (suma del PSS).")
    Age: str = Field(..., example="20-25", description="Edad del estudiante como rango (ej. '20-25', '26-30').")
    Gender: str = Field(..., example="Femenino", description="Género del estudiante (Femenino, Masculino, Otro).")
    Academic_Year: str = Field(..., example="Segundo Año o Equivalente", description="Año académico del estudiante (ej. 'Primer Año o Equivalente').")
    Current_CGPA: str = Field(..., example="3.00 - 3.39", description="Categoría del CGPA actual (ej. 'Less than 2.00', '3.70 - 4.00', 'Below 2.50').")

class UserCreate(BaseModel):
    email: str
    username: str
    password: str
    role: Optional[str] = "student"

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    role: str
    model_config = {
        "from_attributes": True # Configuración para compatibilidad con ORM
    }

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class SleepEntryCreate(BaseModel):
    hours: float
    quality: str
    bedtime_routine: Optional[str] = None
    feeling_upon_waking: str

    class Config:
        from_attributes = True

class SleepEntrySchema(SleepEntryCreate):
    id: int
    user_id: int
    recorded_at: datetime

    class Config:
        from_attributes = True

# --- ENDPOINTS ---

@app.get("/")
async def read_root():
    return {"message": "Bienvenido al API del Modelo Predictivo de Riesgo en Salud Mental Universitaria"}

@app.post("/predict", summary="Predice el nivel de riesgo en salud mental", response_model=dict)
async def predict(request_data: PredictRequest, current_user: User = Depends(get_current_user)):
    print(f"Predicción solicitada por usuario: {current_user.email} (ID: {current_user.id}, Rol: {current_user.role})")
    
    if model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Modelo de predicción no cargado o no disponible.")

    try:
        features_for_model_df = preprocess_for_prediction(request_data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error al preprocesar los datos: {str(e)}")

    try:
        prediction_code = model.predict(features_for_model_df)[0]
        prediction_proba = model.predict_proba(features_for_model_df)[0]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al realizar la predicción: {str(e)}")

    risk_labels = {0: 'Bajo/Mínimo', 1: 'Moderado', 2: 'Alto/Severo'}
    prediction_label = risk_labels.get(int(prediction_code), 'Desconocido')

    if prediction_code >= 1:
        print(f"ALERTA GENERADA: Riesgo '{prediction_label}' detectado para {current_user.email}.")
    
    recommendations_list = []
    if prediction_code == 0:
        recommendations_list = [
            "Mantén hábitos saludables: buena alimentación, ejercicio regular, sueño adecuado.",
            "Sigue buscando actividades que te generen placer y te ayuden a relajarte.",
            "Cultiva tus relaciones sociales y busca apoyo en amigos y familiares.",
            "Considera practicar técnicas de mindfulness o meditación."
        ]
    elif prediction_code == 1:
        recommendations_list = [
            "Habla con un amigo, familiar o mentor de confianza sobre cómo te sientes.",
            "Considera buscar orientación o apoyo psicológico en los servicios de bienestar de tu universidad o externos.",
            "Identifica las fuentes de estrés y busca estrategias para manejarlas.",
            "Asegúrate de tomar descansos adecuados del estudio y otras actividades."
        ]
    elif prediction_code == 2:
        recommendations_list = [
            "Contacta a un profesional de la salud mental (psicólogo o psiquiatra) lo antes posible. No lo pospongas.",
            "Comunícaselo a alguien de confianza (familiar, amigo, profesor) y pídeles ayuda de inmediato.",
            "Evita aislarte. Mantente conectado con tus seres queridos y busca apoyo.",
            "Recuerda que buscar ayuda es un signo de fortaleza y es el primer paso hacia la recuperación. No estás solo/a."
        ]
        recommendations_list.append("Si sientes que estás en una crisis o en peligro inminente, por favor, contacta a una línea de ayuda para crisis o emergencia en tu país de inmediato.")

    return {
        "predicted_risk_level_code": int(prediction_code),
        "predicted_risk_level_label": prediction_label,
        "raw_scores_received": {
            "depression_total_score": request_data.Depression_Score,
            "anxiety_total_score": request_data.Anxiety_Score,
            "stress_total_score": request_data.Stress_Score
        },
        "probabilities": {
            "Bajo_Minimo": float(prediction_proba[0]),
            "Moderado": float(prediction_proba[1]),
            "Alto_Severo": float(prediction_proba[2])
        },
        "recommendations": recommendations_list
    }

@app.post("/register", response_model=UserResponse, summary="Registra un nuevo usuario")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user_email = db.query(User).filter(User.email == user.email).first()
    if db_user_email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El correo electrónico ya está registrado.")
    db_user_username = db.query(User).filter(User.username == user.username).first()
    if db_user_username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El nombre de usuario ya está en uso.")

    hashed_password = get_password_hash(user.password)
    new_user = User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        role=user.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/token", response_model=Token, summary="Obtiene un token de acceso JWT")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user_by_email_or_username(db, identifier=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=UserResponse, summary="Obtiene información del usuario actual")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# --- NUEVOS ENDPOINTS PARA EL DIARIO DE SUEÑO ---

@app.post("/sleep-tracker/", response_model=SleepEntrySchema, summary="Registra un nuevo hábito de sueño")
async def create_sleep_entry(
    sleep_entry: SleepEntryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user) # Usuario autenticado
):
    db_sleep_entry = SleepEntry(
        **sleep_entry.model_dump(),
        user_id=current_user.id
    )
    db.add(db_sleep_entry)
    db.commit()
    db.refresh(db_sleep_entry)
    return db_sleep_entry

@app.get("/sleep-tracker/", response_model=List[SleepEntrySchema], summary="Obtiene el historial de hábitos de sueño del usuario")
async def get_sleep_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user) # Usuario autenticado
):
    sleep_entries = db.query(SleepEntry).filter(SleepEntry.user_id == current_user.id).order_by(SleepEntry.recorded_at.desc()).all()
    return sleep_entries