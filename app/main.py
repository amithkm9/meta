from fastapi import FastAPI

from app.routes import router

app = FastAPI(
    title="SignAdapt",
    description="Adaptive Sign-Language Tutoring Environment",
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    from app.config import APP_HOST, APP_PORT

    uvicorn.run("app.main:app", host=APP_HOST, port=APP_PORT, reload=False)
