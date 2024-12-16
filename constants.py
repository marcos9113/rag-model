GEMINI_API_KEY = "AIzaSyAVh_1Nl55D1HU9GJQpaGWFhwOiYtdZx6g" 
PDF_FILE_PATH = "training_data"
UNIVERSAL_PROMPT = """
I will provide a prompt and a list of contexts ordered by their similarity to the prompt. Lower order indicates higher similarity.
If no relevant context is available or you're unsure, provide a generalized response, clearly stating it's based on lack of context.

PROMPT:
{{prompt}}

CONTEXT:
{{context}}

Respond in a beautiful markdown response without any additional text. Respond with the most accurate answer you can generate from the context

"""


