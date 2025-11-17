"""
FastAPI Backend for SODAI Drinks Prediction System
Provides endpoints for predictions and recommendations.
"""
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_loader import get_model_loader
from predictor import Predictor
from recommender import Recommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
model_loader = None
predictor = None
recommender = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI app."""
    global model_loader, predictor, recommender

    # Startup: Initialize model loader (but don't require model to exist yet)
    logger.info("Starting up application...")
    try:
        model_loader = get_model_loader()

        # Try to load model, but don't fail if it doesn't exist
        try:
            model = model_loader.get_model()
            logger.info("✓ Model loaded successfully on startup")

            # Initialize predictor and recommender
            predictor = Predictor(model=model)
            recommender = Recommender(model=model)
            logger.info("✓ Predictor and Recommender initialized")

        except Exception as e:
            logger.warning(f"⚠ Model not available on startup: {e}")
            logger.warning("Application will start anyway. Model will be loaded on first request.")
            logger.warning("Please ensure the Airflow DAG has been executed to train the model.")

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application...")


def ensure_model_loaded():
    """
    Ensure model is loaded. Initialize predictor and recommender if needed.
    Raises HTTPException if model cannot be loaded.
    """
    global predictor, recommender

    if model_loader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model loader not initialized"
        )

    # If predictor and recommender already initialized, we're good
    if predictor is not None and recommender is not None:
        return

    # Try to load model
    try:
        logger.info("Lazy loading model on first request...")
        model = model_loader.get_model()

        # Initialize predictor and recommender
        predictor = Predictor(model=model)
        recommender = Recommender(model=model)

        logger.info("✓ Model, predictor and recommender initialized successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available. Please ensure the Airflow DAG has been executed to train the model. Error: {str(e)}"
        )


# Create FastAPI app
app = FastAPI(
    title="SODAI Drinks Prediction API",
    description="API for predicting customer purchase behavior and generating product recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class PredictRequest(BaseModel):
    """Request model for single prediction."""
    customer_id: int = Field(..., description="Customer ID", example=1001)
    product_id: int = Field(..., description="Product ID", example=2001)


class PredictResponse(BaseModel):
    """Response model for single prediction."""
    customer_id: int
    product_id: int
    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    probability: float = Field(..., description="Purchase probability (0-1)")
    week: int
    customer_type: str
    product_brand: str
    product_category: str


class RecommendRequest(BaseModel):
    """Request model for recommendations."""
    customer_id: int = Field(..., description="Customer ID", example=1001)
    top_n: int = Field(5, description="Number of recommendations", ge=1, le=20)


class Recommendation(BaseModel):
    """Model for a single recommendation."""
    rank: int
    product_id: int
    probability: float
    brand: str
    category: str
    sub_category: str
    segment: str
    package: str
    size: float


class RecommendResponse(BaseModel):
    """Response model for recommendations."""
    customer_id: int
    recommendations: List[Recommendation]
    total_recommendations: int


class ModelInfo(BaseModel):
    """Model information."""
    source: str = Field(..., description="Model source (mlflow or local)")
    run_id: Optional[str] = None
    val_recall: Optional[float] = None
    val_precision: Optional[float] = None
    val_f1: Optional[float] = None
    val_auc_pr: Optional[float] = None
    path: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_info: ModelInfo


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "SODAI Drinks Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns application status and model information.
    """
    try:
        model_loaded = model_loader is not None and model_loader.model is not None
        model_info_dict = model_loader.get_model_info() if model_loaded else {}

        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            model_info=ModelInfo(**model_info_dict) if model_info_dict else ModelInfo(source="none")
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get information about the currently loaded model.
    """
    if model_loader is None or model_loader.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        model_info_dict = model_loader.get_model_info()
        return ModelInfo(**model_info_dict)
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}"
        )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """
    Reload the model (useful after retraining).
    """
    global predictor, recommender

    try:
        logger.info("Reloading model...")
        model = model_loader.reload_model()

        # Reinitialize predictor and recommender
        predictor = Predictor(model=model)
        recommender = Recommender(model=model)

        return {
            "message": "Model reloaded successfully",
            "model_info": model_loader.get_model_info()
        }
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Make a purchase prediction for a customer-product pair.

    Returns the probability that the customer will purchase the product
    in the next week, along with a binary prediction (0 or 1).
    """
    # Ensure model is loaded (lazy loading)
    ensure_model_loaded()

    try:
        logger.info(f"Prediction request: customer={request.customer_id}, product={request.product_id}")
        result = predictor.predict(
            customer_id=request.customer_id,
            product_id=request.product_id
        )
        return PredictResponse(**result)

    except ValueError as e:
        # Customer or product not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendation"])
async def recommend(request: RecommendRequest):
    """
    Generate top N product recommendations for a customer.

    Returns a list of products ordered by purchase probability,
    with the most likely purchases at the top.
    """
    # Ensure model is loaded (lazy loading)
    ensure_model_loaded()

    try:
        logger.info(f"Recommendation request: customer={request.customer_id}, top_n={request.top_n}")
        recommendations = recommender.recommend(
            customer_id=request.customer_id,
            top_n=request.top_n
        )

        return RecommendResponse(
            customer_id=request.customer_id,
            recommendations=[Recommendation(**rec) for rec in recommendations],
            total_recommendations=len(recommendations)
        )

    except ValueError as e:
        # Customer not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation failed: {str(e)}"
        )


@app.get("/customers/sample", tags=["Utility"])
async def get_sample_customers(limit: int = 10):
    """
    Get a sample of customer IDs for testing.
    """
    # Ensure model is loaded (lazy loading)
    ensure_model_loaded()

    try:
        customers = predictor.get_available_customers(limit=limit)
        return {"customers": customers, "count": len(customers)}
    except Exception as e:
        logger.error(f"Failed to get customers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve customers: {str(e)}"
        )


@app.get("/products/sample", tags=["Utility"])
async def get_sample_products(limit: int = 10):
    """
    Get a sample of product IDs for testing.
    """
    # Ensure model is loaded (lazy loading)
    ensure_model_loaded()

    try:
        products = predictor.get_available_products(limit=limit)
        return {"products": products, "count": len(products)}
    except Exception as e:
        logger.error(f"Failed to get products: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve products: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
