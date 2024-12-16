import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class GeminiChat:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = self.api_key
        self.text_llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.image_llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")

    def process_text(self, prompt: str):
        result = self.text_llm.invoke(prompt)
        return result.content

    def process_image(self, image_url: str, additional_text: str = None):
        message_content = []
        if additional_text:
            message_content.append({"type": "text", "text": additional_text})
        message_content.append({"type": "image_url", "image_url": image_url})

        message = HumanMessage(content=message_content)
        result = self.image_llm.invoke([message])
        return result