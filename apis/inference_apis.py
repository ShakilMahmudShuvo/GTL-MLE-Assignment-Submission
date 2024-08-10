from fastapi import HTTPException
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from services.inference_service.model_inference import InferenceService
from services.inference_service.onnx_model_infer import ONNXInferenceService

router = APIRouter()

@router.post("/infer")
async def infer_text(text: str, model_type: str, inference_service: InferenceService = Depends()):
    if model_type == "torch":
        result = inference_service.infer(text)
    elif model_type == "onnx":
        result = inference_service.infer(text)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'torch' or 'onnx'")
    return JSONResponse(result)


