''' python
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- 1. Document Loading ---
# For this example, we'll use a simple string as our document.
# In a real-world scenario, you would load your documents from files (PDF, TXT, etc.)
document = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially
criticized by some of France's leading artists and intellectuals for its design, but it has
become a global cultural icon of France and one of the most recognizable structures in the world.
"""

# --- 2. Document Splitting ---
# We split the document into smaller chunks to make it easier to process.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(document)
print("--- Document Chunks ---")
for i, text in enumerate(texts):
    print(f"Chunk {i+1}: {text}\n")

# --- 3. Embedding Model ---
# We use a pre-trained model from Hugging Face to convert our text chunks into numerical vectors (embeddings).
# These embeddings capture the semantic meaning of the text.
print("--- Loading Embedding Model ---")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding Model Loaded.\n")

# --- 4. Vector Store ---
# We use FAISS (Facebook AI Similarity Search) to create a vector store.
# This store allows for efficient searching of the most similar text chunks to a given query.
print("--- Creating Vector Store ---")
db = FAISS.from_texts(texts, embeddings)
print("Vector Store Created.\n")


# --- 5. Retriever ---
# The retriever's job is to fetch the most relevant document chunks from the vector store based on the user's query.
retriever = db.as_retriever()

# --- 6. Language Model and QA Pipeline ---
# We'll use a pre-trained question-answering model from Hugging Face.
# This model will take the user's question and the retrieved document chunks to generate an answer.
print("--- Loading Language Model and QA Pipeline ---")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
)
# We need to wrap our pipeline in a LangChain-compatible format.
llm = HuggingFacePipeline(pipeline=qa_pipeline)
print("Language Model and QA Pipeline Loaded.\n")
 
# --- 7. RAG Chain ---
# We'll use LangChain Expression Language (LCEL) to create a modern RAG chain.
# This chain will perform the entire RAG process:
# 1. Take the user's question.
# 2. Use the retriever to find relevant document chunks.
# 3. Create a prompt with the question and the retrieved context.
# 4. Pass the prompt to the language model.
# 5. Parse the output.
 
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)
 
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
 
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt_custom
    | llm
    | StrOutputParser()
)
 
# --- 8. Querying the Pipeline ---
def answer_question(question):
    """
    This function takes a user's question, queries the RAG pipeline, and prints the answer.
    """
    print(f"Query: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}\n")

print("--- Querying the RAG Pipeline ---")
answer_question("Where is the Eiffel Tower located?")
answer_question("Who designed the Eiffel Tower?")
answer_question("When was the Eiffel Tower built?")
answer_question("Why was the Eiffel Tower built?")

print("--- RAG Pipeline Example Complete ---")
'''