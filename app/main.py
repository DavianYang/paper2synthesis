import pickle
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import pdfplumber
import re
from contextlib import asynccontextmanager

app = FastAPI()

# Initialize model variable at the global level
loaded_model = None

# Use lifespan to load the model at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model
    try:
        # Load the model at startup
        with open('./models/svm_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        print("Model loaded successfully.")
        yield
    except Exception as e:
        print(f"Error loading model: {e}")
    finally:
        print("Shutdown process complete.")
        # Any cleanup can go here (if needed)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with a specific domain like ["http://localhost:5500"] for better security
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

@app.post("/predict/")
async def extract_abstract(uploaded_file: UploadFile = File(...)):
    if loaded_model is None:
        return {"error": "Model is not loaded yet. Please try again."}

    try:
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

        # Use the extracted abstract for prediction
        prediction_input = abstract if abstract else full_text
        prediction = loaded_model.predict([prediction_input])  # Pass the extracted text as input
        
        print('Prediction:', prediction)

        return {"result": prediction[0]}  # Adjust as necessary for your model output
    except Exception as e:
        return {"error": str(e)}
