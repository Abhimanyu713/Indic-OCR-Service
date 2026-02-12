import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from ocr_service import ocr_service
from schemas import OCRResponse, ErrorResponse

app = FastAPI(
    title="Multilingual OCR API",
    description="High-accuracy OCR using FastAPI, LangChain, and OpenAI Vision",
    version="1.0.0"
)

ALLOWED_EXTENSIONS = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

@app.post(
    "/v1/ocr",
    response_model=OCRResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def perform_ocr(file: UploadFile = File(...)):
    """
    Upload an image to extract text using GPT-4o Vision.
    Supports English and major Indian languages.
    """
    # Validate File Type
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Supported: JPEG, PNG, WebP"
        )

    try:
        #Read file content
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file provided."
            )

        #Process with LangChain Service
        result = await ocr_service.process_image(image_data, file.content_type)
        return result

    except RuntimeError as re:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(re)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during processing."
        )
    finally:
        await file.close()

@app.get("/health")
async def health_check():
    return {"status": "online"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)