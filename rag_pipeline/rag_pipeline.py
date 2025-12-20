import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_community.llms import Ollama
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

def get_initial_documents():
    """Returns the initial documents to seed the vector store."""
    return ["""
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially
    criticized by some of France's leading artists and intellectuals for its design, but it has
    become a global cultural icon of France and one of the most recognizable structures in the world.
    """]

def split_documents(documents):
    """Splits a list of documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents(documents)
    return texts

def split_text(text_content):
    """Splits a single string of text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text_content)

def load_embedding_model():
    """Loads the sentence transformer embedding model from Hugging Face."""
    print("--- Loading Embedding Model ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding Model Loaded.\n")
    return embeddings

def load_or_create_vector_store(embeddings):
    """Loads a FAISS vector store from disk, or creates it if it doesn't exist."""
    if os.path.exists(FAISS_INDEX_PATH):
        print("--- Loading Vector Store from Disk ---")
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector Store Loaded.\n")
    else:
        print("--- Creating and Saving Vector Store ---")
        initial_docs = get_initial_documents()
        texts = split_documents(initial_docs)
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(FAISS_INDEX_PATH)
        print("Vector Store Created and Saved.\n")
    return db

def load_hf_llm():
    """Loads the LLM and QA pipeline from Hugging Face."""
    print("--- Loading Language Model and QA Pipeline ---")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    qa_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    print("Language Model and QA Pipeline Loaded.\n")
    return llm

def load_ollama_llm(model_name: str, base_url: str = "http://localhost:11434"):
    """Loads an LLM from a local Ollama instance."""
    print(f"--- Loading Ollama Model: {model_name} from {base_url} ---")
    llm = Ollama(model=model_name, base_url=base_url)
    print("Ollama Model Loaded.\n")
    return llm

def create_rag_chain(llm, retriever):
    """Creates a RAG chain that returns a dictionary with 'answer' and 'context'."""
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Helpful Answer:"""
    rag_prompt_custom = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_for_answer = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )

    chain = RunnableParallel(
        {"answer": rag_chain_for_answer, "context": retriever}
    )

    return chain

def main():
    """Main function to run the RAG pipeline as a script."""
    # 1. Load components
    embeddings = load_embedding_model()
    db = load_or_create_vector_store(embeddings)
    llm = load_hf_llm()

    # 2. Create retriever and RAG chain
    retriever = db.as_retriever()
    rag_chain = create_rag_chain(llm, retriever)

    # 3. Define a query function
    def answer_question(question):
        """
        This function takes a user's question, queries the RAG pipeline, and prints the answer.
        """
        print(f"Query: {question}")
        result = rag_chain.invoke(question)
        print(f"Answer: {result['answer']}\n")
        print("--- CONTEXT ---")
        for doc in result['context']:
            print(doc)
            print("-" * 20)

    # 4. Run example queries
    print("--- Querying the RAG Pipeline ---")
    answer_question("Where is the Eiffel Tower located?")
    answer_question("Who designed the Eiffel Tower?")
    answer_question("When was the Eiffel Tower built?")
    answer_question("Why was the Eiffel Tower built?")
    print("--- RAG Pipeline Example Complete ---")

if __name__ == "__main__":
    main()
