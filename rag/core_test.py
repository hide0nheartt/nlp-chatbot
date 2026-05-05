import os
import re
import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# 1. Load biến môi trường

load_dotenv()

# Nếu file .env không có DEFAULT_SEARCH_LIMIT thì mặc định lấy 10
DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", 10))

# 2. Tạo embedding function (tiếng việt)

# Dùng model tiếng Việt để biến query/document thành vector
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="keepitreal/vietnamese-sbert"
)

# Class RAG
class RAG:
    def __init__(self, collection1_name: str, collection2_name: str, db_path: str):
        # Khởi tạo database
        self.client = chromadb.PersistentClient(path=db_path)

        # Collection 1: title + description (semantic mạnh)
        self.collection1 = self.client.get_or_create_collection(
            name=collection1_name,
            embedding_function=sentence_transformer_ef
        )

        # Collection 2: title + category (lọc loại)
        self.collection2 = self.client.get_or_create_collection(
            name=collection2_name,
            embedding_function=sentence_transformer_ef
        )

    # 3. Tiền xử lý query (text)
    def preprocess_text(self, text: str) -> str:
        # Nếu text rỗng thì trả về chuỗi rỗng
        if not text:
            return ""
        # Chuyển về chữ thường
        text = str(text).lower()
        # Xóa ký tự đặc biệt
        text = re.sub(r"[^\w\s,.]", "", text)
        # Xóa khoảng trắng thừa
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # 4. Parse kết quả từ ChromaDB
    def parse_chroma_results(self, results):
        # Chuyển kết quả ChromaDB thành list dict dễ dùng hơn
        parsed = []

        if not results["ids"] or not results["ids"][0]:
            return parsed

        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]

            parsed.append({
                "_id": results["ids"][0][i],
                "url": meta.get("url", ""),
                "title": meta.get("title", ""),
                "description": meta.get("description", ""),
                "price": meta.get("price", ""),
                "image_url": meta.get("image_url", ""),
                "category": meta.get("category", ""),
                "distance": results["distances"][0][i]
            })

        return parsed

    # 5. Weighted Reciprocal Rank Fusion
    def weighted_reciprocal_rank(self, doc_lists, weights=None, c=60):
        # Gộp nhiều danh sách kết quả bằng RRF
        if weights is None:
            weights = [1] * len(doc_lists)

        if len(doc_lists) != len(weights):
            raise ValueError("Số lượng doc_lists phải bằng số lượng weights.")

        rrf_scores = {}
        doc_map = {}

        for doc_list, weight in zip(doc_lists, weights):
            for rank, doc in enumerate(doc_list, start=1):
                # Dùng _id/url làm khóa để tránh trùng mô tả
                doc_id = doc["_id"]

                # Công thức RRF
                score = weight * (1 / (rank + c))

                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                    doc_map[doc_id] = doc

                rrf_scores[doc_id] += score

        # Sắp xếp document theo điểm RRF giảm dần
        sorted_ids = sorted(rrf_scores, key = rrf_scores.get, reverse=True)

        return [doc_map[i] for i in sorted_ids]

    # 6. Search trong 1 collection
    def search_one(self, query, collection, limit):
        # Query gốc
        res1 = collection.query(
            query_texts=[query],
            n_results=limit
        )

        # Query đã tiền xử lý
        clean_query = self.preprocess_text(query)

        res2 = collection.query(
            query_texts=[clean_query],
            n_results=limit
        )

        parsed1 = self.parse_chroma_results(res1)
        parsed2 = self.parse_chroma_results(res2)

        return self.weighted_reciprocal_rank([parsed1, parsed2])

    # 7. Hybrid search nâng cấp (2 collection)
    def hybrid_search(self, query: str, limit=DEFAULT_SEARCH_LIMIT):
        if not query.strip():
            return []

        # Search collection 1: title + description
        results1 = self.search_one(query, self.collection1, limit)

        # Search collection 2: title + category
        results2 = self.search_one(query, self.collection2,limit)

        # Gộp kết quả 2 collection bằng RRF (ưu tiên semantic (title +description))
        final_results = self.weighted_reciprocal_rank(
            doc_lists=[results1, results2],
            weights=[2, 1]
        )

        return final_results[:limit]

    # 8. Tạo context cho LLM
    def enhance_prompt(self, query: str, limit=DEFAULT_SEARCH_LIMIT):
        # Lấy kqua liên quan từ vector database
        results = self.hybrid_search(query, limit)

        if not results:
            return "Không tìm thấy sản phẩm phù hợp."

        prompt = ""

        for i, r in enumerate(results, 1):
            prompt += (
                f"[Sản phẩm {i}]\n"
                f"Tên: {r['title']}\n"
                f"Giá: {r['price']}\n"
                f"Danh mục: {r['category']}\n"
                f"Mô tả: {r['description']}\n"
                f"Link: {r['url']}\n\n"
                f"Hình ảnh: {r['image_url']}\n\n"
    
            )

        return prompt

    # 9. In kết quả search
    def display_results(self, results):
        if not results:
            print("Không tìm thấy sản phẩm phù hợp.")
            return

        for i, r in enumerate(results, 1):
            print(f"Sản phẩm {i}:")
            print("Tên:", r["title"])
            print("Giá:", r["price"])
            print("Danh mục:", r["category"])
            print("Mô tả:", r["description"])
            print("Hình ảnh:", r["image_url"])
            print("Link:", r["url"])
            print("-" * 50)


# 10. Test nhanh
if __name__ == "__main__":
    rag = RAG(
        "chewy_chewy_aihi_01",
        "chewy_chewy_aihi_02",
        "VECTOR_STORE"
    )

    query = "tôi muốn mua bánh sinh nhật cho bạn gái"

    results = rag.hybrid_search(query)

    rag.display_results(results)

    print("\n    PROMPT     \n")
    print(rag.enhance_prompt(query))