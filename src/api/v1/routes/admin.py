from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
from src.ingestion.ingestion import ingest_file

router = APIRouter()

@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")
    
    os.makedirs("temp_uploads", exist_ok=True)
    temp_path = os.path.join("temp_uploads", file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        ingest_file(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest file: {str(e)}")
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return {"message": "File uploaded and vectorized successfully", "filename": file.filename}
