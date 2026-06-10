from fastapi import APIRouter, Response

router = APIRouter()

router.get("/health")


def health_check():
    return Response("OK")
