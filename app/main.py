from fastapi import FastAPI

from app.routes import router

app = FastAPI(
    title="SignAdapt",
    description=(
        "OpenEnv-compliant adaptive sign-language tutoring environment. "
        "An AI agent plans step-by-step teaching interventions for a simulated "
        "deaf/hard-of-hearing learner whose comprehension, attention, frustration, "
        "and confidence evolve dynamically. The learner's comprehension is hidden "
        "and can only be revealed through assessment actions. Grading is outcome-based: "
        "did the learner actually improve?"
    ),
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    from app.config import APP_HOST, APP_PORT

    uvicorn.run("app.main:app", host=APP_HOST, port=APP_PORT, reload=False)
