from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.health import router as health_router
from api.routes.predict import router as predict_router
from src.pipeline import InferencePipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = InferencePipeline()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(predict_router)
app.include_router(health_router)
