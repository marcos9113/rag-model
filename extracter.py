import fitz
import docx
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ContextExtractor:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        self.texts_metadata = self.extract_text_from_files(dir_path)
        
        self.chunks, self.metadata = self.split_text_into_chunks(self.texts_metadata)
        
        self.embeddings = self.create_embeddings(self.chunks)
        
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)

    def extract_text_from_files(self, dir_path):
        texts_metadata = []
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if filename.endswith('.pdf'):
                texts_metadata.extend(self.extract_text_from_pdf(file_path, filename))
            elif filename.endswith('.docx'):
                texts_metadata.extend(self.extract_text_from_docx(file_path, filename))
            elif filename.endswith('.txt'):
                texts_metadata.extend(self.extract_text_from_txt(file_path, filename))
        return texts_metadata

    def extract_text_from_pdf(self, pdf_path, filename):
        doc = fitz.open(pdf_path)
        texts_metadata = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            texts_metadata.append((text, filename, page_num + 1))
        return texts_metadata

    def extract_text_from_docx(self, docx_path, filename):
        doc = docx.Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return [(text, filename, None)]

    def extract_text_from_txt(self, txt_path, filename):
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [(text, filename, None)]

    def split_text_into_chunks(self, texts_metadata, chunk_size=256):
        chunks = []
        metadata = []
        for text, filename, page_num in texts_metadata:
            words = text.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)
                metadata.append((filename, page_num, i // chunk_size))
        return chunks, metadata

    def create_embeddings(self, chunks):
        embeddings = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return np.array(embeddings)

    def get_context(self, question, k=1):
        question_embedding = self.create_embeddings([question])
        distances, indices = self.index.search(question_embedding, k)
        results = []
        for i in indices[0]:
            chunk = self.chunks[i]
            meta = self.metadata[i]
            results.append((chunk, meta))
        return results

# # Usage example
# dir_path = "your_directory_path"
# extractor = ContextExtractor(dir_path)
# context_results = extractor.get_context("your query here", k=1)
# for context, (filename, page_num, chunk_index) in context_results:
#     print(f"Context: {context}")
#     print(f"Filename: {filename}")
#     if page_num:
#         print(f"Page number: {page_num}")
#     else:
#         print("Page number: N/A (not a PDF)")
#     print(f"Chunk index: {chunk_index}")
