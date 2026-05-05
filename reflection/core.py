import json
import uuid
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_ROLE_MAPPING = {
    "human": "user",
    "ai": "assistant"
}

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="keepitreal/vietnamese-sbert"
)


class Reflection:
    def __init__(
        self,
        llm,
        db_path: str,
        dbChatHistoryCollection: str,
        semanticCacheCollection: str,
    ):
        self.client = chromadb.PersistentClient(path=db_path)

        self.his_collection = self.client.get_or_create_collection(
            name=dbChatHistoryCollection,
            embedding_function=sentence_transformer_ef
        )

        self.semantic_cache_collection = self.client.get_or_create_collection(
            name=semanticCacheCollection,
            embedding_function=sentence_transformer_ef
        )

        self.llm = llm
        self.dbChatHistoryCollection = dbChatHistoryCollection

    def chat(
        self,
        session_id: str,
        enhanced_message: str,
        original_message: str = "",
        cache_response: bool = False,
        query_embedding: list = None
    ):
        system_prompt_content = """Bạn là chatbot của tiệm bánh Chewy Chewy. Vai trò của bạn là hỗ trợ khách hàng tìm hiểu về các sản phẩm của cửa hàng như bánh sinh nhật, bánh su, set bánh mini và các sản phẩm liên quan.

Bạn cần:
1. Trả lời nhanh chóng, chính xác, dùng xưng hô "Mình" và "bạn".
2. Tư vấn sản phẩm dựa trên thông tin được cung cấp từ hệ thống RAG.
3. Nếu khách hỏi về sản phẩm, hãy nêu tên sản phẩm, giá, mô tả ngắn và gợi ý phù hợp.
4. Nếu khách chỉ chào hỏi hoặc trò chuyện bình thường, hãy trả lời thân thiện, lịch sự.
5. Không tự bịa thông tin sản phẩm nếu không có trong dữ liệu.
6. Không nói về hoa/cây cảnh vì cửa hàng là tiệm bánh.

Hãy giữ giọng văn thân thiện, chuyên nghiệp và dễ hiểu."""

        messages = [
            {"role": "system", "content": system_prompt_content},
            *self.__construct_session_messages__(
                self.his_collection.get(where_document={"$contains": session_id})
            ),
            {"role": "user", "content": enhanced_message}
        ]

        response = self.llm.chat(messages)

        if not response:
            return "Xin lỗi, hiện tại hệ thống AI local đang gặp lỗi. Bạn vui lòng thử lại sau."

        self.__record_human_prompt__(session_id, enhanced_message, original_message)
        self.__record_ai_response__(session_id, response)

        if cache_response and query_embedding:
            self.__cache_ai_response__(
                enhanced_message=enhanced_message,
                original_message=original_message,
                response=response,
                query_embedding=query_embedding
            )

        return response

    def __construct_session_messages__(self, session_messages: dict):
        result = []

        if not session_messages["ids"]:
            return result

        for session_message in session_messages["documents"]:
            session_message = json.loads(session_message)

            result.append({
                "role": OPEN_AI_ROLE_MAPPING[session_message["History"]["type"]],
                "content": session_message["History"]["data"]["content"]
            })

        return result

    def __record_human_prompt__(
        self,
        session_id: str,
        enhanced_message: str,
        original_message: str
    ):
        self.his_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[
                json.dumps({
                    "SessionId": session_id,
                    "History": {
                        "type": "human",
                        "data": {
                            "type": "human",
                            "content": original_message,
                            "enhanced_content": enhanced_message,
                            "additional_kwargs": {},
                            "response_metadata": {},
                            "name": None,
                            "id": None,
                        }
                    }
                }, ensure_ascii=False)
            ],
        )

    def __record_ai_response__(self, session_id: str, response: str):
        self.his_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[
                json.dumps({
                    "SessionId": session_id,
                    "History": {
                        "type": "ai",
                        "data": {
                            "type": "ai",
                            "content": response,
                            "enhanced_content": None,
                            "additional_kwargs": {},
                            "name": None,
                            "id": str(uuid.uuid4()),
                            "usage_metadata": {},
                            "response_metadata": {
                                "model_name": "llama3.2",
                                "provider": "ollama"
                            },
                        }
                    }
                }, ensure_ascii=False)
            ],
        )

    def __cache_ai_response__(
        self,
        enhanced_message: str,
        original_message: str,
        response: str,
        query_embedding: list
    ):
        self.semantic_cache_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[query_embedding],
            documents=[
                json.dumps({
                    "text": [
                        {
                            "type": "human",
                            "content": original_message,
                            "enhanced_content": enhanced_message,
                            "additional_kwargs": {},
                            "response_metadata": {},
                            "name": None,
                            "id": None,
                        }
                    ],
                    "llm_string": {
                        "model_name": "llama3.2",
                        "name": "Ollama"
                    },
                    "return_val": [
                        {
                            "type": "ai",
                            "content": response,
                            "enhanced_content": None,
                            "additional_kwargs": {},
                            "name": None,
                            "id": str(uuid.uuid4()),
                            "usage_metadata": {},
                            "response_metadata": {
                                "model_name": "llama3.2",
                                "provider": "ollama"
                            },
                        }
                    ]
                }, ensure_ascii=False)
            ]
        )