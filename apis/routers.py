from fastapi import APIRouter
from . import inference_apis, start_training_api

api_router = APIRouter()

api_router.include_router(inference_apis.router, tags=["inference"])
api_router.include_router(start_training_api.router, tags=["training"])
