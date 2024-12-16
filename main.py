from flask import Flask, render_template, request, jsonify
from constants import GEMINI_API_KEY, PDF_FILE_PATH, UNIVERSAL_PROMPT
from gemini import GeminiChat
import os
from extracter import ContextExtractor

app = Flask(__name__)
gemini_chat = GeminiChat(api_key=GEMINI_API_KEY)
pdf_context_extracter = ContextExtractor(dir_path=PDF_FILE_PATH)

def get_response(question):
    contexts = pdf_context_extracter.get_context(question=question, k=3)
    context_text = {}
    metadata_info = {}
    for index, item in enumerate(contexts):
        filename, page_num = item[1][0],item[1][1]
        order_key = f"Order {index+1}"
        context_text[order_key] = item[0]
        metadata_info[filename] = page_num
    prompt = UNIVERSAL_PROMPT.replace("{{context}}", "\n\n".join(context_text.values())).replace("{{prompt}}", question)
    print(prompt)
    response = gemini_chat.process_text(prompt=prompt)
    data = {
        "response": response,
        "context": context_text,
        "metadata": metadata_info
    }
    return data

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    data = get_response(user_input)
    generated_response = {
        "generated_response": data["response"],
        "metadata": data["metadata"]
    }
    return jsonify(generated_response)

if __name__ == "__main__":
    app.run(debug=True)