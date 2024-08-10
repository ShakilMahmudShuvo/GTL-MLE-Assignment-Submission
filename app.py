import os
import uvicorn
import warnings
from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from apis.routers import api_router
from apis.routers import api_router
from utils.custom_logger import logger

warnings.filterwarnings("ignore")


def include_router(app):
    app.include_router(api_router)


app = FastAPI(
    title="POS and NER Model API",
)

include_router(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5723)
