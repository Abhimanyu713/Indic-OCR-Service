import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import settings
from schemas import OCRResponse

class OCRService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            openai_api_key=settings.openai_api_key,
            max_tokens=settings.max_tokens
        )
        self.parser = JsonOutputParser(pydantic_object=OCRResponse)

    def _encode_image(self, image_bytes: bytes) -> str:
        """Converts raw bytes to base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def process_image(self, image_bytes: bytes, mime_type: str) -> OCRResponse:
        """
        Processes image via LangChain and OpenAI Vision.
        Supports Multilingual extraction (English, Hindi, Bengali, etc.)
        """
        base64_image = self._encode_image(image_bytes)

        # Prompt instruction ensuring JSON output and multilingual support
        prompt_text = (
            "You are an expert OCR engine. Extract all text from the provided image. "
            "Maintain the original formatting where possible. "
            "The document may contain English or Indian languages (Hindi, Marathi, etc.). "
            "Return the output strictly in JSON format matching the schema."
        )

        # Constructing the message for OpenAI Vision
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"{prompt_text}\n{self.parser.get_format_instructions()}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                },
            ]
        )

        # LangChain Runnable sequence
        chain = self.llm | self.parser
        
        try:
            response = await chain.ainvoke([message])
            return response
        except Exception as e:
            raise RuntimeError(f"LangChain processing failed: {str(e)}")

# Global instance for dependency injection/reuse
ocr_service = OCRService()