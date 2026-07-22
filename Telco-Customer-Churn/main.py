from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import shap
import io
import logging

# إعداد الـ Logging الاحترافي للشركات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnterpriseInferenceAPI")

app = FastAPI(
    title="🛡️ Enterprise MLOps Inference Engine", 
    version="2.0.0",
    description="Production-Grade Churn Pipeline with Decoupled XAI Layer and Strict Input Validation."
)

# --- 1. API Contract & Validation Layer (Pydantic Models) ---
class ModelMetadata(BaseModel):
    model_name: str
    version: str
    environment: str

class ClientRecord(BaseModel):
    customerID: str
    Risk_Level: str
    Churn_Prob: float = Field(..., description="Calculated probability from XGBoost Model")
    Expected_Annual_Loss: float
    MonthlyCharges: float

class InferenceResponse(BaseModel):
    metadata: ModelMetadata
    total_exposure: float
    critical_accounts_count: int
    dynamic_high_cutoff_pct: float
    records: List[ClientRecord]

class ShapFeatureImpact(BaseModel):
    feature: str
    shap_value: float

class ShapAuditResponse(BaseModel):
    customerID: str
    version: str
    drivers: List[ShapFeatureImpact]


# --- 2. Model Registry & Caching Architecture ---
MODEL_REGISTRY = {
    "v2.0.4": {"model_path": r"C:\Users\ASUS\Documents\up_work\Telco-Customer-Churn\final_xgb_model.pkl", "preprocessor_path":          r"C:\Users\ASUS\Documents\up_work\Telco-Customer-Churn\preprocessor.pkl"}
    }
ACTIVE_VERSION = "v2.0.4"

class ModelContainer:
    """Lazy loading and Caching mechanism for production memory management"""
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.feature_names = None

    def load_pipeline(self, version: str):
        logger.info(f"Initializing Artifact Registry for version: {version}")
        try:
            self.model = joblib.load(MODEL_REGISTRY[version]["model_path"])
            self.preprocessor = joblib.load(MODEL_REGISTRY[version]["preprocessor_path"])
            self.explainer = shap.TreeExplainer(self.model)
            self.feature_names = self.preprocessor.get_feature_names_out().tolist()
            logger.info("All model artifacts mapped successfully to memory.")
        except Exception as e:
            logger.error(f"Critical Registry Failure: {str(e)}")
            raise RuntimeError(f"Failed to load engine artifacts: {str(e)}")

# إقلاع وحفظ الموديلات في الـ Memory لمرة واحدة فقط عند تشغيل السيرفر
engine = ModelContainer()
engine.load_pipeline(ACTIVE_VERSION)


# --- 3. Async Inference Core Endpoints ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "active_version": ACTIVE_VERSION}

@app.post("/predict", response_model=InferenceResponse, status_code=status.HTTP_200_OK)
async def predict_churn_batch(file: UploadFile = File(...)):
    """Optimized Real-time Inference Pipeline with zero SHAP overhead during execution"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Unsupported File Type. Only structured CSV files accepted.")
    
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=422, detail="Unprocessable Entity. CSV payload is corrupted.")

    # التحقق من وجود الأعمدة الأساسية المطلوبة للـ Feature Engineering والتدريب
    required_columns = {'tenure', 'MonthlyCharges', 'TotalCharges'}
    if not required_columns.issubset(df.columns):
        raise HTTPException(status_code=400, detail=f"Data Validation Schema Failure. Missing columns: {required_columns - set(df.columns)}")

    # تنظيف البيانات وعزل الـ Production Environment عن الـ Exceptions
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna().copy()
    
    if df.empty:
        raise HTTPException(status_code=422, detail="No valid data rows found after dropping missing records.")

    # Live Pipeline Feature Engineering
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['IsNewCustomer'] = (df['tenure'] < 12).astype(int)
    df['HighChargeFlag'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
    
    drop_cols = ['customerID', 'Churn']
    X_inference = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # تحويل البيانات واستدعاء الـ XGBoost
    try:
        X_processed = engine.preprocessor.transform(X_inference)
        probabilities = engine.model.predict_proba(X_processed)[:, 1]
    except Exception as pipeline_err:
        logger.error(f"Transformation Breakdown: {str(pipeline_err)}")
        raise HTTPException(status_code=500, detail="Internal Pipeline Transformation Failure.")

    df['Churn_Prob'] = probabilities
    
    # Dynamic Thresholds (حساب مستويات الخطورة بناءً على توزيع البيانات المرفوعة)
    high_cutoff = np.percentile(probabilities, 80)
    medium_cutoff = np.percentile(probabilities, 50)
    
    df['Risk_Level'] = np.where(df['Churn_Prob'] >= high_cutoff, '🔴 High Risk', 
                                np.where(df['Churn_Prob'] >= medium_cutoff, '🟡 Medium Risk', '🟢 Low Risk'))
    
    df['Expected_Annual_Loss'] = (df['Churn_Prob'] * df['MonthlyCharges'] * 12).round(2)
    
    # صياغة الـ Records بناءً على الـ Strict Pydantic Contracts
    records_payload = df[['customerID', 'Risk_Level', 'Churn_Prob', 'Expected_Annual_Loss', 'MonthlyCharges']].to_dict(orient="records")
    
    return InferenceResponse(
        metadata=ModelMetadata(model_name=MODEL_REGISTRY[ACTIVE_VERSION]["model_path"], version=ACTIVE_VERSION, environment="Production"),
        total_exposure=float(df['Expected_Annual_Loss'].sum()),
        critical_accounts_count=int(len(df[df['Churn_Prob'] >= high_cutoff])),
        dynamic_high_cutoff_pct=float(high_cutoff * 100),
        records=records_payload
    )

@app.post("/audit-customer", response_model=ShapAuditResponse, status_code=status.HTTP_200_OK)
async def audit_single_customer_xai(customer_id: str, file: UploadFile = File(...)):
    """Isolated, Decoupled XAI On-Demand Endpoint. Solves memory spike issues by auditing single clients on demand."""
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    
    if 'customerID' not in df.columns or customer_id not in df['customerID'].astype(str).values:
        raise HTTPException(status_code=404, detail=f"Client '{customer_id}' not found in the provided telemetry block.")
        
    # استخراج السطر الخاص بالعميل المختار وعمل الـ Engineering له بمفرده
    client_row = df[df['customerID'].astype(str) == customer_id].copy()
    client_row['AvgMonthlySpend'] = client_row['TotalCharges'] / (client_row['tenure'] + 1)
    client_row['IsNewCustomer'] = (client_row['tenure'] < 12).astype(int)
    client_row['HighChargeFlag'] = (client_row['MonthlyCharges'] > client_row['MonthlyCharges'].median()).astype(int)
    
    X_single = client_row.drop(columns=['customerID', 'Churn'], errors='ignore')
    X_single_processed = engine.preprocessor.transform(X_single)
    
    # حساب الـ SHAP لهذ العميل الفردي فقط (سرعة استجابة فائقة وحماية للـ Memory)
    shap_values = engine.explainer.shap_values(X_single_processed)
    
    top_indices = np.argsort(np.abs(shap_values[0]))[::-1][:4] # جلب أعلى 4 ميزات مسببة للقرار
    
    drivers_list = [
        ShapFeatureImpact(feature=engine.feature_names[idx], shap_value=float(shap_values[0][idx]))
        for idx in top_indices
    ]
    
    return ShapAuditResponse(customerID=customer_id, version=ACTIVE_VERSION, drivers=drivers_list)