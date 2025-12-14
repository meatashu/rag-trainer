import streamlit as st
import numpy as np
from rag_pipeline import rag_pipeline as rag
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os,requests
import tempfile
from streamlit_agraph import agraph, Node, Edge, Config
from collections import Counter
import ollama
import re

# --- App Configuration ---
st.set_page_config(
    page_title="Knowledge Base Manager",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Knowledge Base Interaction & Management")
st.markdown("""
Welcome to the RAG Pipeline dashboard. Use the tabs below to interact with the knowledge base.
""")

# --- State Management ---
if "pending_docs" not in st.session_state:
    st.session_state.pending_docs = []

# --- Load Core RAG Components ---
@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model."""
    embeddings = rag.load_embedding_model()
    return embeddings

embeddings = load_embedding_model()

# --- Sidebar for LLM Configuration ---
st.sidebar.title("⚙️ LLM Configuration")
use_ollama = st.sidebar.toggle("Use Ollama LLM", value=False)

# Add a toggle for debug mode
debug_mode = st.sidebar.toggle("Enable Debug Logging", value=False)
if debug_mode:
    st.sidebar.info("Debug logging is enabled. View the console for detailed logs of the RAG chain.")


if use_ollama:
    ollama_base_url = st.sidebar.text_input("Ollama Server URL", value="http://localhost:11434")
    if not ollama_base_url.startswith("http"):
        st.sidebar.error("Please enter a valid URL starting with http:// or https://")
        st.stop()

    try:
        # Explicitly create a client to connect to the specified server
        client = ollama.Client(host=ollama_base_url)
        # This single line correctly fetches the list of available models
        ollama_models = [m["model"] for m in client.list()["models"]]
        for model in client.list()["models"]:
            print(f"- Model Name: {model['model']}, Size: {model['size']}, Modified At: {model['modified_at']}, details: {model['details']['parameter_size']}, {model['details']['quantization_level']}, Family: {model['details']['family']}")

        # Set a default model if it exists, otherwise default to the first model
        default_model = "mistral:latest"
        default_index = ollama_models.index(default_model) if default_model in ollama_models else 0

        selected_model = st.sidebar.selectbox(
            "Select an Ollama Model",
            options=ollama_models,
            index=default_index
        )
        llm = rag.load_ollama_llm(selected_model, base_url=ollama_base_url)
        st.sidebar.success(f"Connected to Ollama at {ollama_base_url}!\nUsing model: {selected_model}")
    except Exception as e:
        st.sidebar.error(f"Could not connect to Ollama server at {ollama_base_url}: {e}")
        st.stop()
else:
    # Cache the Hugging Face LLM when not using Ollama
    @st.cache_resource
    def get_hf_llm():
        return rag.load_hf_llm()
    llm = get_hf_llm()
    st.sidebar.info("Using default Hugging Face model (google/flan-t5-base).")


# Load the vector store. This is not cached as it can be updated.
db = rag.load_or_create_vector_store(embeddings)

# --- UI Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "❓ RAG QA",
    "📚 View Knowledge",
    "➕ Add Knowledge",
    "✅ Approve Knowledge"
])

# --- TAB 1: RAG QA & Similarity ---
with tab1:
    st.header("Query the Knowledge Base")
    st.write("Ask questions or enter a phrase to find the most relevant information in the vector store.")

    query = st.text_input("Enter your question or search phrase:", key="qa_query")

    if query:
        with st.spinner("Searching for relevant documents..."):
            # Get retriever and format docs
            rag_used = False
            retriever = db.as_retriever()
            similar_docs = retriever.invoke(query)

            st.subheader("Most Relevant Chunks:")
            if similar_docs:
                for i, doc in enumerate(similar_docs):
                    with st.expander(f"Chunk {i+1} (Similarity Score: {doc.metadata.get('score', 'N/A')})"):
                        st.write(doc.page_content)
                rag_used = True
            else:
                st.warning("No relevant documents found.")

        st.subheader("Answer from RAG Pipeline:")
        with st.spinner("Generating answer..."):
            if rag_used:
                st.info("💡 Answer augmented by the knowledge base (RAG).", icon="🧠")
            else:
                st.info("💡 Answering directly with the LLM (no relevant knowledge found).", icon="🤖")
            rag_chain = rag.create_rag_chain(llm, retriever)
            answer = rag_chain.invoke(query)
            st.write(answer)

# --- TAB 2: View Existing Knowledge ---
with tab2:
    st.header("Browse Existing Knowledge")
    st.write("These are the documents currently indexed in the FAISS vector store.")

    if st.button("Refresh Knowledge View"):
        # Re-load the vector store to get the latest data
        db = rag.load_or_create_vector_store(embeddings)
        st.rerun()

    if db and db.docstore:
        all_docs = db.docstore._dict
        if not all_docs:
            st.info("The knowledge base is currently empty.")
        else:
            # --- Knowledge Graph Visualization ---
            st.subheader("Interactive Knowledge Graph")
            if st.toggle("Generate Knowledge Graph", value=False):
                with st.spinner("Building graph..."):
                    nodes = []
                    edges = []
                    all_keywords = []

                    # Simple stop words list
                    stop_words = set(["the", "a", "an", "in", "on", "is", "it", "and", "to", "of", "for", "was", "were"])

                    for doc_id, doc in all_docs.items():
                        # Find words, lowercase them, and filter out stop words and short words
                        words = re.findall(r'\b\w+\b', doc.page_content.lower())
                        keywords = [word for word in words if word not in stop_words and len(word) > 3]
                        all_keywords.extend(keywords)
                        
                        # Create edges for co-occurring keywords in the same doc
                        for i in range(len(keywords)):
                            for j in range(i + 1, len(keywords)):
                                if keywords[i] != keywords[j]: # Avoid self-loops
                                    edges.append(Edge(source=keywords[i], target=keywords[j]))

                    # Count keyword frequency to size nodes
                    keyword_counts = Counter(all_keywords)
                    
                    # Create unique nodes with size based on frequency
                    for keyword, count in keyword_counts.items():
                        nodes.append(Node(id=keyword, 
                                          label=keyword, 
                                          size=10 + count * 2, # Base size + increment per occurrence
                                          shape="dot") 
                                    )

                    # Configure the graph layout and physics for better interaction
                    config = Config(width=750,
                                    height=600,
                                    directed=False,
                                    # Let the physics engine run live for a more dynamic view
                                    physics={'barnesHut': {'gravitationalConstant': -8000, 'springConstant': 0.04, 'springLength': 250}},
                                    interaction={'dragNodes':True, 'dragView': True, 'zoomView': True},
                                    # Hierarchical layout can be an alternative
                                    hierarchical=False,
                                    )

                    agraph(nodes=nodes, edges=edges, config=config)

            # --- Raw Knowledge Chunk Viewer ---
            st.subheader("Browse and Manage Knowledge Chunks")
            st.write(f"Found **{len(all_docs)}** document chunks.")
            for i, (doc_id, doc) in enumerate(all_docs.items()):
                with st.expander(f"Chunk {i+1} (ID: {doc_id[:8]}...)"):
                    st.text(doc.page_content)
                    if st.button("Delete Chunk", key=f"delete_{doc_id}", type="primary"):
                        # The delete method expects a list of IDs
                        success = db.delete([doc_id])
                        db.save_local(rag.FAISS_INDEX_PATH)
                        st.success(f"Chunk {doc_id} deleted successfully!")
                        st.rerun()
    else:
        st.warning("Could not retrieve documents from the vector store.")

# --- TAB 3: Add New Knowledge ---
with tab3:
    st.header("Add New Knowledge")
    st.write("Add new information via text, document upload, or speech-to-text.")

    # Option 1: Text Input
    st.subheader("Add by Text")
    text_input = st.text_area("Paste or write your text here:")
    if st.button("Stage Text for Approval"):
        if text_input:
            st.session_state.pending_docs.append({"content": text_input, "source": "Text Input"})
            st.success("Text has been staged for approval.")
        else:
            st.warning("Please enter some text.")

    # Option 2: Document Upload
    st.subheader("Add by Document Upload")
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    if st.button("Stage Document for Approval"):
        if uploaded_file is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                loader = TextLoader(tmp_file_path) if uploaded_file.type == "text/plain" else PyPDFLoader(tmp_file_path)
                docs = loader.load()
                content = "\n\n".join([doc.page_content for doc in docs])
                st.session_state.pending_docs.append({"content": content, "source": uploaded_file.name})
                st.success(f"Document '{uploaded_file.name}' has been staged for approval.")
                os.remove(tmp_file_path) # Clean up temp file
            except Exception as e:
                st.error(f"Failed to process file: {e}")
        else:
            st.warning("Please upload a file.")

    # Option 3: Speech-to-Text (Placeholder)
    st.subheader("Add by Speech (Future Feature)")
    st.info("Functionality to add knowledge via speech-to-text can be implemented here.")

# --- TAB 4: Approve Knowledge ---
with tab4:
    st.header("Approve Pending Knowledge")
    st.write("Review and approve new information to add it to the main knowledge base.")

    if not st.session_state.pending_docs:
        st.info("There are no pending documents to approve.")
    else:
        st.write(f"You have **{len(st.session_state.pending_docs)}** documents pending approval.")

        for i, doc_info in enumerate(st.session_state.pending_docs):
            with st.expander(f"Pending Document {i+1} (Source: {doc_info['source']})"):
                st.text(doc_info['content'])

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Approve", key=f"approve_{i}"):
                        with st.spinner("Processing and adding to knowledge base..."):
                            try:
                                # Split, embed, and add the new text
                                new_text_chunks = rag.split_text(doc_info['content'])
                                db.add_texts(new_text_chunks)
                                db.save_local(rag.FAISS_INDEX_PATH)

                                # Remove from pending list
                                st.session_state.pending_docs.pop(i)
                                st.success("Knowledge approved and added to the vector store!")
                                st.rerun() # Rerun to update the UI
                            except Exception as e:
                                st.error(f"Failed to add knowledge: {e}")
                with col2:
                    if st.button("Reject", key=f"reject_{i}"):
                        st.session_state.pending_docs.pop(i)
                        st.warning("Knowledge rejected.")
                        st.rerun() # Rerun to update the UI

    if st.session_state.pending_docs and st.button("Clear All Pending Documents"):
        st.session_state.pending_docs = []
        st.success("All pending documents have been cleared.")
        st.rerun()