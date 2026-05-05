import unittest
from semantic_router import SemanticRouter, Route
from semantic_router.samples import productSample, chitchatSample

class RouterTestCase(unittest.TestCase):
    def setUp(self):
        self.PRODUCT_ROUTE_NAME = "products"
        self.CHITCHAT_ROUTE_NAME = "chitchat"

        productRoute = Route(name=self.PRODUCT_ROUTE_NAME, samples=productSample)
        chitchatRoute = Route(name=self.CHITCHAT_ROUTE_NAME, samples=chitchatSample)
        self.semanticRouter = SemanticRouter(routes=[productRoute, chitchatRoute])

    def test_chitchat_route(self):
        chitchat_queries = [
            "chào bạn",
            "bạn có khỏe không?",
            "hello shop",
            "cảm ơn bạn nhé",
            "tạm biệt",
        ]

        for query in chitchat_queries:
            print(query, "=>", self.semanticRouter.guide(query))
            self.assertEqual(
                self.semanticRouter.guide(query)[1],
                self.CHITCHAT_ROUTE_NAME,
                "incorrect semantic route for chitchat query"
            )

    def test_products_route(self):
        products_queries = [
            "Bạn có bánh sinh nhật nào không?",
            "Tôi muốn mua set bánh mini",
            "Có bánh su vị chocolate không?",
            "Giá EVENT CAKE S02 bao nhiêu?",
            "Tôi muốn mua bánh tặng sinh nhật bạn gái",
            "Shop gợi ý giúp tôi vài loại bánh ngon",
        ]

        for query in products_queries:
            print(query, "=>", self.semanticRouter.guide(query))
            self.assertEqual(
                self.semanticRouter.guide(query)[1],
                self.PRODUCT_ROUTE_NAME,
                "incorrect semantic route for products query"
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)