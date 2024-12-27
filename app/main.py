from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import pdfplumber
import re
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with a specific domain like ["http://localhost:5500"] for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

loaded_model = joblib.load('./models/svm_model.pkl')

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
async def extract_abstract(uploaded_file: UploadFile):

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

        result = prediction[0].item()
        print(result)
        return {"result": result}  # Adjust as necessary for your model output
    except Exception as e:
        return {"error": str(e)}
