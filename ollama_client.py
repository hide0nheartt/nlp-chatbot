import requests


class OllamaClient:
    def __init__(self, api_key=None):
        self.url = "http://localhost:11434/api/chat"
        self.model = "llama3.2"

    def chat(self, messages):
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )

            data = response.json()

            return data["message"]["content"]

        except Exception as e:
            print("Ollama error:", e)
            return "Xin lỗi, AI local đang lỗi."