# Streamlit Web Interface for Your Medical Chatbot
# Creates a professional web app interface

import streamlit as st
import sys
import os
from datetime import datetime
import json

# Import your existing chatbot (assuming it's in the same directory)
# You'll need to copy your FixedConversationalChatbot class here or import it

# First, let's recreate the essential parts of your working chatbot for the web interface
import re
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict
from collections import defaultdict

class WebMedicalChatbot:
    def __init__(self):
        """Simplified version of your working chatbot for web interface"""
        # Setup components
        self.setup_embeddings()
        self.setup_vector_database()
        
        # Medical knowledge
        self.knowledge_chunks = []
        self.medical_keywords = self.setup_medical_keywords()
        
        # Conversation tracking for web interface
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
    @st.cache_resource
    def setup_embeddings(_self):
        """Setup embedding model with Streamlit caching"""
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            return embedding_model, True
        except Exception as e:
            st.error(f"Embeddings failed: {e}")
            return None, False
    
    def setup_vector_database(self):
        """Setup ChromaDB for web"""
        try:
            self.chroma_client = chromadb.PersistentClient(path="./web_medical_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="web_pediatric",
                metadata={"description": "Web pediatric medical knowledge"}
            )
            return True
        except Exception as e:
            st.error(f"Database failed: {e}")
            return False
    
    def setup_medical_keywords(self):
        """Setup medical keyword categories"""
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
        """Load medical data for web interface"""
        if uploaded_file:
            # Handle uploaded file
            content = uploaded_file.read().decode('utf-8')
        elif file_path:
            # Handle file path
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except FileNotFoundError:
                st.error(f"File {file_path} not found!")
                return False
        else:
            return False
        
        # Create chunks (simplified version of your working method)
        chunks = self.create_smart_chunks(content)
        self.knowledge_chunks = chunks
        
        # Generate embeddings
        embedding_model, use_embeddings = self.setup_embeddings()
        
        if use_embeddings and embedding_model:
            try:
                # Clear existing
                try:
                    self.collection.delete(where={})
                except:
                    pass
                
                # Process in batches
                batch_size = 20
                progress_bar = st.progress(0)
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    texts = [chunk['text'] for chunk in batch]
                    embeddings = embedding_model.encode(texts)
                    ids = [f"chunk_{j}" for j in range(i, i+len(batch))]
                    
                    self.collection.add(
                        embeddings=embeddings.tolist(),
                        documents=texts,
                        ids=ids
                    )
                    
                    # Update progress
                    progress = min((i + batch_size) / len(chunks), 1.0)
                    progress_bar.progress(progress)
                
                progress_bar.empty()
                return True
                
            except Exception as e:
                st.error(f"Embeddings failed: {e}")
                return False
        
        return True
    
    def create_smart_chunks(self, text: str) -> List[Dict]:
        """Create smart chunks (simplified version)"""
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
        """Identify chunk topic"""
        text_lower = text.lower()
        
        for symptom, keywords in self.medical_keywords['symptoms'].items():
            if any(keyword in text_lower for keyword in keywords):
                return symptom
        
        return 'general'
    
    def chat(self, query: str) -> str:
        """Main chat function for web interface"""
        if not query.strip():
            return "Please ask me about pediatric health topics!"
        
        # Handle greetings
        if self.handle_greetings(query):
            return self.handle_greetings(query)
        
        if not self.knowledge_chunks:
            return "Please upload your medical data file first!"
        
        # Find relevant information
        relevant_chunks = self.find_relevant_chunks(query)
        
        # Create response
        if relevant_chunks:
            response = self.create_response(relevant_chunks, query)
        else:
            response = "I don't have specific information about this topic in my pediatric database. Please consult with your child's healthcare provider."
        
        return response
    
    def handle_greetings(self, query: str) -> str:
        """Handle simple greetings"""
        query_lower = query.lower().strip()
        
        if query_lower in ['hello', 'hi', 'hey']:
            return "👋 Hello! I'm here to help with pediatric medical questions. What would you like to know about your child's health?"
        
        if query_lower in ['thank you', 'thanks']:
            return "😊 You're welcome! Feel free to ask more pediatric health questions."
        
        if query_lower in ['bye', 'goodbye']:
            return "👋 Goodbye! Take care, and feel free to return with any pediatric health questions."
        
        return None
    
    def find_relevant_chunks(self, query: str, n_results: int = 3) -> List[str]:
        """Find relevant chunks"""
        embedding_model, use_embeddings = self.setup_embeddings()
        
        if use_embeddings and embedding_model:
            try:
                query_embedding = embedding_model.encode([query])
                results = self.collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=n_results
                )
                
                if results['documents'] and results['documents'][0]:
                    return results['documents'][0]
            except Exception:
                pass
        
        # Fallback to keyword search
        return self.keyword_search(query, n_results)
    
    def keyword_search(self, query: str, n_results: int) -> List[str]:
        """Keyword-based search fallback"""
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
        """Create medical response"""
        main_content = chunks[0]
        sentences = [s.strip() for s in main_content.split('.') if len(s.strip()) > 20]
        
        # Take first 2 sentences
        if len(sentences) >= 2:
            response = '. '.join(sentences[:2]) + '.'
        else:
            response = sentences[0] + '.' if sentences else main_content[:300] + '...'
        
        # Add medical disclaimer
        response += "\n\n⚠️ **Medical Disclaimer:** This information is for educational purposes only. Always consult your child's healthcare provider for personalized medical advice."
        
        return response

# Streamlit Web Interface
def main():
    # Page configuration
    st.set_page_config(
        page_title="Pediatric Medical Chatbot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: right;
    }
    .bot-message {
        background-color: #e9ecef;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Pediatric Medical Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info("""
        This is an AI-powered pediatric medical assistant that can help answer questions about children's health.
        
        **Features:**
        • Evidence-based medical information
        • Conversational interface
        • Educational medical guidance
        """)
        
        st.header("⚠️ Important Notice")
        st.warning("""
        **This is not a substitute for professional medical advice.**
        
        Always consult with qualified healthcare professionals for:
        • Medical emergencies
        • Diagnosis and treatment
        • Personalized medical advice
        """)
        
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your medical data file",
            type=['txt'],
            help="Upload your Pediatric_cleaned.txt file"
        )
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = WebMedicalChatbot()
        st.session_state.data_loaded = False
    
    # Handle file upload
    if uploaded_file and not st.session_state.data_loaded:
        with st.spinner("Loading medical data..."):
            success = st.session_state.chatbot.load_medical_data(uploaded_file=uploaded_file)
            if success:
                st.session_state.data_loaded = True
                st.success("✅ Medical data loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to load medical data")
    
    # Main chat interface
    if st.session_state.data_loaded:
        st.subheader("💬 Chat with the Medical Assistant")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("Conversation History")
            for i, exchange in enumerate(st.session_state.conversation_history):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**You:** {exchange['user']}")
                        st.markdown(f"**Assistant:** {exchange['bot']}")
                    with col2:
                        st.caption(exchange['timestamp'])
                st.divider()
        
        # Chat input
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask your pediatric medical question:",
                    placeholder="e.g., What should I do if my child has a fever?",
                    key="user_input"
                )
            
            with col2:
                send_button = st.button("Send 📤", type="primary")
        
        # Process user input
        if send_button and user_input:
            # Get response from chatbot
            with st.spinner("Getting medical information..."):
                response = st.session_state.chatbot.chat(user_input)
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                'user': user_input,
                'bot': response,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            # Clear input and rerun to show new conversation
            st.rerun()
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What causes fever in children?",
            "How should I treat my child's cough?",
            "When should I call the doctor?",
            "What are signs of dehydration in children?",
            "How can I prevent my child from getting sick?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(question, key=f"sample_{i}"):
                    # Simulate clicking on sample question
                    st.session_state.user_input = question
                    response = st.session_state.chatbot.chat(question)
                    st.session_state.conversation_history.append({
                        'user': question,
                        'bot': response,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                    st.rerun()
    
    else:
        # Instructions for first-time users
        st.subheader("🚀 Getting Started")
        st.info("""
        **Welcome to the Pediatric Medical Chatbot!**
        
        To get started:
        1. 📁 Upload your medical data file (Pediatric_cleaned.txt) using the sidebar
        2. 💬 Start asking questions about pediatric health
        3. 🔄 Have a conversation with the AI assistant
        
        The chatbot will provide evidence-based information from your medical database.
        """)
        
        # Demo mode option
        st.subheader("🎭 Demo Mode")
        if st.button("Load Sample Data for Demo"):
            # You can add sample data loading here
            st.info("Demo mode would load sample medical data. Upload your actual data file for full functionality.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏥 Pediatric Medical Chatbot | Educational Medical Information</p>
        <p><small>Always consult healthcare professionals for medical advice</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()