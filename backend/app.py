import os
from fastapi import FastAPI, File, UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import Base, Receipt
from backend.ocr_utils import extract_text_from_image
from backend.parser_utils import parse

UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

engine = create_engine("sqlite:///receipts.db", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded file
    with open(path, "wb") as f:
        f.write(await file.read())

    size = os.path.getsize(path)
    print("\n============================")
    print("File saved at:", path)
    print("File size:", size, "bytes")
    print("============================\n")

    # OCR
    text = extract_text_from_image(path)

    # Load ML classifier
    from backend.ml_classifier import ReceiptClassifier
    classifier = ReceiptClassifier()
    classifier.load()

    # CORRECT way to predict
    category = classifier.predict(text)[0]

    print("Predicted Category:", category)

    # Parse fields
    data = parse(text)

    # OVERRIDE parser category with ML category
    data["category"] = category
    data["raw_text"] = text  # include full OCR for frontend

    # Save to DB
    db = Session()
    receipt = Receipt(**data, filename=file.filename)
    db.add(receipt)
    db.commit()

    return {"status": "ok", "data": data}
