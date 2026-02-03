from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.Pipeline.predict_pipeline import CustomData, PredictPipeline

# Create FastAPI app
app = FastAPI()

# Templates folder
templates = Jinja2Templates(directory="templates")

# Load model and preprocessor once globally
predict_pipeline = PredictPipeline()


# Home route: GET displays form, POST predicts
@app.get("/", response_class=HTMLResponse)
async def index_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def index_post(
    request: Request,
    gender: str = Form(...),
    ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...)
):
    # Prepare input data
    data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )

    pred_df = data.get_data_as_data_frame()
    results = predict_pipeline.predict(pred_df)

    # Return the same template with results
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results[0]
        }
    )
