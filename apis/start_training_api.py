from fastapi import APIRouter, HTTPException
from services.trainer_base import TrainerBase
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/start_training")
async def start_training():
    try:
        trainer = TrainerBase()
        metrics = trainer.run()
        return JSONResponse(metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
