from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from gradio_client import Client
from PIL import Image
import pytesseract
import pdfplumber
import re
import joblib

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/template")

space_url = "https://huggingface.co/spaces/DavianYang/evidence-synthesis/api/predict"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

def extract_abstract_from_text(text: str) -> str:
    """
    Extracts the entire abstract section from the given text using regex.
    Captures everything from 'Abstract' until the next section heading.
    """
    match = re.search(
        r"(?:Abstract|ABSTRACT|abstract)\s*[:\-]?\s*(.*?)(?=\n\s*(?:[A-Za-z ]{2,})(?:\n|:|\s{2,}))", 
        text, 
        re.DOTALL
    )
    return match.group(1).strip() if match else "Abstract section not found."

def read_file(uploaded_file):
    # Extract text from PDF or Image
    if uploaded_file.filename.endswith(".pdf"):
        # Process PDFs
        with pdfplumber.open(uploaded_file.file) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
            abstract = extract_abstract_from_text(full_text)
    else:
        # Process Images
        image = Image.open(uploaded_file.file)
        full_text = pytesseract.image_to_string(image)
        abstract = extract_abstract_from_text(full_text)

    return abstract if abstract else full_text


def predict(model_path, uploaded_file):
    prediction_input = read_file(uploaded_file)
    model = joblib.load(model_path)
    prediction = model.predict([prediction_input])

    return prediction

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_binary/")
async def svm_binary(uploaded_file: UploadFile):
    try:
        prediction = predict('./models/svm_model.pkl', uploaded_file)
        return {"result": prediction[0].item()}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_multilabel/")
async def svm_multilabel(uploaded_file: UploadFile):
    try:
        prediction = predict('./models/svm_multi_model.pkl', uploaded_file)
        return {"result": [i for i, _ in enumerate(prediction.squeeze(0).tolist()) if i == 1]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_bert")
async def bert_binary(uploaded_file: UploadFile):
    try:
        prediction_input = read_file(uploaded_file)

        client = Client("DavianYang/evidence-synthesis")
        result = client.predict(text=prediction_input, api_name="/predict")
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}