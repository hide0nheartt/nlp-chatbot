from flask import Flask, request, jsonify
from flask_cors import CORS
from ollama_client import OllamaClient
from rag import RAG
from semantic_router import SemanticRouter, Route
from semantic_router.samples import productSample, chitchatSample
from reflection import Reflection
from embedding_model import EmbeddingModel
import os
from dotenv import load_dotenv

load_dotenv()

# Load biến môi trường
db_chat_history_collection = os.getenv("DB_CHAT_HISTORY_COLLECTION", "chewy_chewy_chat_history")
semantic_cache_collection = os.getenv("semanticCacheCollection", "chewy_chewy_semantic_cache")

db_path = "VECTOR_STORE"

# Khởi tạo Flask
app = Flask(__name__)
CORS(app)

# Khởi tạo model / LLM
embedding_model = EmbeddingModel()
llm = OllamaClient()

# Khởi tạo RAG
rag = RAG(
    collection1_name="chewy_chewy_aihi_01",
    collection2_name="chewy_chewy_aihi_02",
    db_path=db_path
)

# Khởi tạo Semantic Router
PRODUCT_ROUTE_NAME = "products"
CHITCHAT_ROUTE_NAME = "chitchat"

productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productSample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)

semanticRouter = SemanticRouter(routes=[productRoute, chitchatRoute])

# Khởi tạo Reflection
reflection = Reflection(
    llm=llm,
    db_path=db_path,
    dbChatHistoryCollection=db_chat_history_collection,
    semanticCacheCollection=semantic_cache_collection
)

@app.route("/api/v1/chewy_chewy", methods=["POST"])
def chat():
    data = request.get_json()

    session_id = data.get("session_id", "default_session")
    query = data.get("query", "")

    if not query.strip():
        return jsonify({
            "role": "assistant",
            "content": "Bạn vui lòng nhập câu hỏi nhé."
        })

    guided_route = semanticRouter.guide(query)[1]
    print(f"semantic route: {guided_route}")

    if guided_route == PRODUCT_ROUTE_NAME:
        query_embedding = embedding_model.get_embedding(query)

        source_information = rag.enhance_prompt(query)

        combined_information = (
            f"Câu hỏi của khách hàng: {query}\n\n"
            f"Hãy trả lời khách hàng dựa trên thông tin sản phẩm sau:\n"
            f"### Sản phẩm ###\n"
            f"{source_information}"
        )

        response = reflection.chat(
            session_id=session_id,
            enhanced_message=combined_information,
            original_message=query,
            cache_response=True,
            query_embedding=query_embedding
        )

    else:
        response = reflection.chat(
            session_id=session_id,
            enhanced_message=query,
            original_message=query,
            cache_response=False
        )

    return jsonify({
        "role": "assistant",
        "content": response
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)