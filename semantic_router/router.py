import numpy as np
from sentence_transformers import SentenceTransformer

# phân loại mục đích người dùng
class SemanticRouter:
    def __init__(self, routes):
        self.routes = routes
        self.embedding_model = SentenceTransformer("keepitreal/vietnamese-sbert") # khởi tạo model, dùng model để biến text thành vector
        self.routesEmbedding = {}
        self.routesEmbeddingCal = {}

        for route in self.routes:
            embeddings = self.embedding_model.encode(route.samples) # encode sample (chữ->số)

            self.routesEmbedding[route.name] = embeddings

            # Chuẩn hóa từng sample riêng biệt
            self.routesEmbeddingCal[route.name] = embeddings / np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True
            )

    def get_routes(self):
        return self.routes

    def guide(self, query):
        queryEmbedding = self.embedding_model.encode([query]) # encode query

        # Chuẩn hóa query (chỉnh vector về độ dài =1)
        queryEmbedding = queryEmbedding / np.linalg.norm(
            queryEmbedding,
            axis=1,
            keepdims=True
        )

        scores = []

        for route in self.routes:
            routeEmbeddingCal = self.routesEmbeddingCal[route.name]

            # Tính cosine similarity giữa query và samples của route (để so sánh query với tất cả sample của route)
            score = np.mean(np.dot(routeEmbeddingCal, queryEmbedding.T).flatten())

            scores.append((score, route.name))

        scores.sort(reverse=True)

        return scores[0]