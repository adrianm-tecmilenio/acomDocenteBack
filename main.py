import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.agent import agent_router

app = FastAPI()

@app.get('/')
def welcome():
    return {'message': 'Bienvenido a acompañamiento para el docente'}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)

