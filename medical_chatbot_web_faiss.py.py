
# Pediatric Medical Chatbot - FAISS Version with Full UI

import streamlit as st
from datetime import datetime
import re
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class WebMedicalChatbot:
    def __init__(self):
        self.embedding_model, self.use_embeddings = self.setup_embeddings()
        self.knowledge_chunks = []
        self.index = None
        self.embedded_texts = []
        self.medical_keywords = self.setup_medical_keywords()

        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

    @st.cache_resource
    def setup_embeddings(_self):
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            return embedding_model, True
        except Exception as e:
            st.error(f"Embeddings failed: {e}")
            return None, False

    def setup_medical_keywords(self):
        return {
            'symptoms': {
                'fever': ['fever', 'temperature', 'hot', 'feverish'],
                'cough': ['cough', 'coughing', 'whooping'],
                'breathing': ['wheezing', 'breathing', 'breath', 'respiratory', 'asthma'],
                'stomach': ['vomiting', 'diarrhea', 'stomach', 'nausea', 'sick'],
                'pain': ['pain', 'hurt', 'ache', 'sore'],
                'rash': ['rash', 'spots', 'red', 'skin']
            },
            'age_groups': {
                'infant': ['infant', 'baby', 'newborn', '0-12 months'],
                'toddler': ['toddler', '1-3 years', 'young child'],
                'child': ['child', 'kid', '4+ years', 'school age']
            },
            'urgency': {
                'emergency': ['emergency', 'urgent', 'serious', 'hospital', 'ambulance'],
                'doctor': ['doctor', 'physician', 'pediatrician', 'call'],
                'home_care': ['home', 'rest', 'fluids', 'monitor']
            }
        }

    def load_medical_data(self, uploaded_file=None, file_path=None):
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
        elif file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except FileNotFoundError:
                st.error(f"File {file_path} not found!")
                return False
        else:
            return False

        chunks = self.create_smart_chunks(content)
        self.knowledge_chunks = chunks

        if self.use_embeddings and self.embedding_model:
            try:
                texts = [chunk['text'] for chunk in chunks]
                embeddings = self.embedding_model.encode(texts)
                self.embedded_texts = texts
                embedding_array = np.array(embeddings).astype("float32")
                dim = embedding_array.shape[1]
                self.index = faiss.IndexFlatL2(dim)
                self.index.add(embedding_array)
                return True
            except Exception as e:
                st.error(f"Embedding/indexing failed: {e}")
                return False
        return True

    def create_smart_chunks(self, text: str) -> List[Dict]:
        chunks = []
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        chunk_id = 0
        for paragraph in paragraphs:
            if len(paragraph.split()) > 200:
                sentences = re.split(r'[.!?]+', paragraph)
                current_chunk = ""
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(current_chunk.split()) + len(sentence.split()) < 150:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append({
                                'id': chunk_id,
                                'text': current_chunk.strip(),
                                'topic': self.identify_chunk_topic(current_chunk)
                            })
                            chunk_id += 1
                        current_chunk = sentence + ". "
                if current_chunk:
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'topic': self.identify_chunk_topic(current_chunk)
                    })
                    chunk_id += 1
            else:
                chunks.append({
                    'id': chunk_id,
                    'text': paragraph,
                    'topic': self.identify_chunk_topic(paragraph)
                })
                chunk_id += 1
        return chunks

    def identify_chunk_topic(self, text: str) -> str:
        text_lower = text.lower()
        for symptom, keywords in self.medical_keywords['symptoms'].items():
            if any(keyword in text_lower for keyword in keywords):
                return symptom
        return 'general'

    def chat(self, query: str) -> str:
        if not query.strip():
            return "Please ask me about pediatric health topics!"
        if self.handle_greetings(query):
            return self.handle_greetings(query)
        if not self.knowledge_chunks:
            return "Please upload your medical data file first!"
        relevant_chunks = self.find_relevant_chunks(query)
        if relevant_chunks:
            response = self.create_response(relevant_chunks, query)
        else:
            response = "I don't have specific information about this topic in my pediatric database. Please consult with your child's healthcare provider."
        return response

    def handle_greetings(self, query: str) -> str:
        query_lower = query.lower().strip()
        if query_lower in ['hello', 'hi', 'hey']:
            return "👋 Hello! I'm here to help with pediatric medical questions. What would you like to know about your child's health?"
        if query_lower in ['thank you', 'thanks']:
            return "😊 You're welcome! Feel free to ask more pediatric health questions."
        if query_lower in ['bye', 'goodbye']:
            return "👋 Goodbye! Take care, and feel free to return with any pediatric health questions."
        return None

    def find_relevant_chunks(self, query: str, n_results: int = 3) -> List[str]:
        if self.use_embeddings and self.embedding_model and self.index:
            try:
                query_embedding = self.embedding_model.encode([query]).astype("float32")
                D, I = self.index.search(np.array(query_embedding), n_results)
                return [self.embedded_texts[i] for i in I[0]]
            except Exception:
                pass
        return self.keyword_search(query, n_results)

    def keyword_search(self, query: str, n_results: int) -> List[str]:
        if not self.knowledge_chunks:
            return []
        query_words = set(query.lower().split())
        chunk_scores = []
        for chunk_info in self.knowledge_chunks:
            chunk_text = chunk_info['text']
            chunk_words = set(chunk_text.lower().split())
            word_overlap = len(query_words.intersection(chunk_words))
            score = word_overlap / len(query_words) if query_words else 0
            if score > 0:
                chunk_scores.append((score, chunk_text))
        chunk_scores.sort(reverse=True)
        return [chunk for _, chunk in chunk_scores[:n_results]]

    def create_response(self, chunks: List[str], query: str) -> str:
        main_content = chunks[0]
        sentences = [s.strip() for s in main_content.split('.') if len(s.strip()) > 20]
        if len(sentences) >= 2:
            response = '. '.join(sentences[:2]) + '.'
        else:
            response = sentences[0] + '.' if sentences else main_content[:300] + '...'
        response += "\n\n⚠️ **Medical Disclaimer:** This information is for educational purposes only. Always consult your child's healthcare provider for personalized medical advice."
        return response

# Streamlit UI
def main():
    st.set_page_config(page_title="Pediatric Medical Chatbot", layout="wide")

    st.title("🏥 Pediatric Medical Chatbot")
    st.markdown("Upload your pediatric medical text file and ask health-related questions.")

    uploaded_file = st.sidebar.file_uploader("📁 Upload Medical Text File", type=["txt"])

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = WebMedicalChatbot()
        st.session_state.data_loaded = False

    if uploaded_file and not st.session_state.data_loaded:
        with st.spinner("Processing medical data..."):
            success = st.session_state.chatbot.load_medical_data(uploaded_file=uploaded_file)
            if success:
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
                st.rerun()

    if st.session_state.data_loaded:
        st.subheader("💬 Ask a Medical Question")
        user_input = st.text_input("Your question:", placeholder="e.g. What are signs of asthma in children?")
        if user_input:
            with st.spinner("Searching medical information..."):
                response = st.session_state.chatbot.chat(user_input)
            st.markdown(f"**🧠 Chatbot:** {response}")

if __name__ == "__main__":
    main()
