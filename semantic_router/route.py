# mục đích: định nghĩa loại câu hỏi of user, giúp router phân loại mục đích của user và quyết định có gọi RAG hay không

from typing import List, Optional

class Route:
    def __init__(self, name: str = None, samples: Optional[List[str]] = None):
        self.name = name
        self.samples = samples or []