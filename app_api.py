from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from jobClassification.pipeline.prediction import PredictionPipeline


text:str = "Job O*NET Classification!!!"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    



@app.post("/predict")
async def predict_route(text):
    try:
        job_title = text["title"]
        job_body = text["body"]
        top_k = text["top_k"]

        obj = PredictionPipeline()
        text = obj.predict(job_title,job_body, top_k)
        return text
    except Exception as e:
        raise e
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)