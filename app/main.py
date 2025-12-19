from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.api import router

app = FastAPI(title="Crypto Fraud Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Explicitly serve index.html on root
from fastapi.responses import FileResponse
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
