from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.malaria import router as malaria_router
from backend.routes.breast_cancer import router as breast_router

app = FastAPI(
    title="Multi Disease Diagnostic API",
    version="1.0.0"
)

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Multi Disease Diagnostic API",
        "docs": "/docs"
    }

app.include_router(
    malaria_router,
    prefix="/malaria",
    tags=["Malaria"]
)

app.include_router(
    breast_router,
    prefix="/breast-cancer",
    tags=["Breast Cancer"]
)