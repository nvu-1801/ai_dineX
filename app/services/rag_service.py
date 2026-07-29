import os
import logging
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")

# Models for Request/Response
class ChatMessage(BaseModel):
    role: str  # "user" or "model"/"assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    branch_id: str = Field(..., alias="branchId")
    session_id: str = Field(..., alias="sessionId")
    menu_context: List[Dict[str, Any]] = Field(..., alias="menuContext")
    chat_history: List[ChatMessage] = Field(..., alias="chatHistory")

    model_config = {
        "populate_by_name": True
    }

    class Config:
        allow_population_by_field_name = True


class OrderItemDraft(BaseModel):
    menu_item_name: str = Field(description="Tên món ăn")
    quantity: int = Field(default=1, description="Số lượng")
    size_text: Optional[str] = Field(default=None, description="Kích thước: small, medium, large")
    toppings: List[str] = Field(default_factory=list, description="Danh sách topping")
    note: Optional[str] = Field(default=None, description="Ghi chú")

class OrderDraft(BaseModel):
    items: List[OrderItemDraft] = Field(default_factory=list)

class ChatAssistantResponse(BaseModel):
    reply: str = Field(description="Câu trả lời của trợ lý ảo bằng Tiếng Việt")
    order_draft: Optional[OrderDraft] = Field(default=None, description="Bản thảo đơn hàng nếu khách đặt món, ngược lại là null")
    recommendations: List[str] = Field(default_factory=list, description="Danh sách tên các món ăn được gợi ý")

class IngestDocument(BaseModel):
    title: str
    content: str

class IngestRequest(BaseModel):
    branch_id: str = Field(..., alias="branchId")
    documents: List[IngestDocument]

    model_config = {
        "populate_by_name": True
    }

    class Config:
        allow_population_by_field_name = True


# Initialize LLM Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
is_mock_mode = not bool(GEMINI_API_KEY)

if not is_mock_mode:
    logger.info("Initializing Google Generative AI client...")
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found in environment. Running in MOCK Mode.")

# Initialize ChromaDB client (local persistent storage)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        if is_mock_mode:
            # Return dummy embeddings of 768 dimensions for mock mode
            return [[0.1] * 768 for _ in input]
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=input,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}. Falling back to mock embeddings.")
            return [[0.1] * 768 for _ in input]

embedding_func = GeminiEmbeddingFunction()

class RagService:
    @staticmethod
    def get_collection(branch_id: str):
        collection_name = f"branch_{branch_id.replace('-', '_')}"
        return chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_func
        )

    async def ingest_documents(self, branch_id: str, docs: List[IngestDocument]) -> bool:
        try:
            collection = self.get_collection(branch_id)
            ids = [f"doc_{branch_id}_{i}" for i in range(len(docs))]
            documents = [d.content for d in docs]
            metadatas = [{"title": d.title} for d in docs]
            
            # Upsert into ChromaDB
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully ingested {len(docs)} documents for branch {branch_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to ingest documents: {e}")
            return False

    async def generate_chat_response(self, request: ChatRequest) -> ChatAssistantResponse:
        # 1. Query policies from VectorDB
        retrieved_policies = []
        try:
            collection = self.get_collection(request.branch_id)
            results = collection.query(
                query_texts=[request.message],
                n_results=2
            )
            if results and results.get("documents") and len(results["documents"][0]) > 0:
                retrieved_policies = results["documents"][0]
                logger.info(f"Retrieved policies: {retrieved_policies}")
        except Exception as e:
            logger.warning(f"Error querying vector DB: {e}. Proceeding without vector context.")

        # 2. Build prompt context
        formatted_menu = ""
        for idx, item in enumerate(request.menu_context):
            name = item.get("name", "Unknown")
            price = item.get("price", "N/A")
            desc = item.get("description", "")
            toppings = ", ".join(item.get("toppings", []))
            formatted_menu += f"- {name}: {price}đ. Mô tả: {desc}. Toppings có sẵn: {toppings}\n"

        context_str = f"=== THỰC ĐƠN CHI NHÁNH ===\n{formatted_menu}\n"
        if retrieved_policies:
            context_str += f"=== CHÍNH SÁCH CHI NHÁNH ===\n" + "\n".join(retrieved_policies) + "\n"

        # 3. Chat with LLM
        if is_mock_mode:
            return self._generate_mock_response(request)

        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=(
                    "Bạn là trợ lý ảo AI của nhà hàng DineX. Hãy trả lời khách hàng một cách thân thiện, lịch sự và chuyên nghiệp bằng Tiếng Việt.\n"
                    "Nhiệm vụ của bạn:\n"
                    "1. Hỗ trợ khách hàng tìm kiếm món ăn và gợi ý thực đơn dựa trên ngữ cảnh thực đơn được cung cấp (menu_context) và các chính sách của cửa hàng (retrieved_policies).\n"
                    "2. Nếu khách hàng có ý định đặt món (ví dụ: 'Cho tôi 1 phở bò chín', 'lấy thêm quẩy', v.v.), hãy phân tích cú pháp tin nhắn và xuất ra đối tượng order_draft có cấu trúc JSON.\n"
                    "3. Nếu không có ý định đặt món, hãy để order_draft là null.\n"
                    "4. KHÔNG tự tiện tính tổng tiền hay báo thanh toán thành công. Chỉ trả về thông tin mặt hàng, số lượng, kích cỡ, topping để app xử lý.\n"
                    "5. Chỉ gợi ý các món ăn thực sự có trong thực đơn. Nếu khách hỏi món không có, hãy trả lời lịch sự là hiện tại chi nhánh này không phục vụ món đó."
                )
            )

            # Format history
            contents = []
            for msg in request.chat_history:
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [msg.content]})

            # Add context to the last prompt message
            prompt_content = f"{context_str}\n=== CÂU HỎI CỦA KHÁCH HÀNG ===\n{request.message}"
            contents.append({"role": "user", "parts": [prompt_content]})

            response = model.generate_content(
                contents=contents,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ChatAssistantResponse,
                    temperature=0.2,
                )
            )
            
            # Parse structured JSON response
            return ChatAssistantResponse.model_validate_json(response.text)
        except Exception as e:
            logger.error(f"Gemini API execution error: {e}. Falling back to mock response.")
            return self._generate_mock_response(request)

    def _generate_mock_response(self, request: ChatRequest) -> ChatAssistantResponse:
        # Mock logic - always return recommendations (dish names) for branch card display
        lowercase_message = request.message.toLowerCase() if hasattr(request.message, 'toLowerCase') else request.message.lower()
        
        reply = "Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?"
        recommendations = []

        # Find matching dishes in menu
        available_dishes = [item.get("name", "") for item in request.menu_context]

        if "bún bò" in lowercase_message or "bun bo" in lowercase_message:
            reply = "Dưới đây là danh sách các chi nhánh của hệ thống DineX đang phục vụ món Bún bò Huế ngon và rẻ nhất cho bạn:"
            recommendations = [d for d in available_dishes if "bún" in d.lower()][:3]
        elif "phở" in lowercase_message or "pho" in lowercase_message:
            reply = "Dưới đây là danh sách các chi nhánh của hệ thống DineX đang phục vụ món Phở bò ngon nhất cho bạn:"
            recommendations = [d for d in available_dishes if "phở" in d.lower()][:3]
        elif "muốn ăn" in lowercase_message or "gợi ý" in lowercase_message:
            reply = "Tôi đề xuất bạn thử các món bán chạy nhất tại các chi nhánh DineX nhé!"
            recommendations = available_dishes[:3]
        else:
            # Try to match any dish name in the message
            matched_dishes = [d for d in available_dishes if d.lower() in lowercase_message]
            if matched_dishes:
                dish = matched_dishes[0]
                reply = f"Dưới đây là danh sách các chi nhánh đang phục vụ món {dish} cho bạn:"
                recommendations = matched_dishes[:3]

        return ChatAssistantResponse(
            reply=reply,
            order_draft=None,
            recommendations=recommendations or available_dishes[:2]
        )

