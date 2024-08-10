from fastapi import APIRouter
from . import inference_apis

api_router = APIRouter()

api_router.include_router(inference_apis.router, tags=["inference"])