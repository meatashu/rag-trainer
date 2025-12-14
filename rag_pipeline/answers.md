
# RAG Pipeline Exercise Answers

Here are the answers to the exercises.

## Exercise 1: Different Data Source

**1. Create `data.txt`:**

Create a file named `data.txt` in the `rag_pipeline` directory with the following content:

```
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially
criticized by some of France's leading artists and intellectuals for its design, but it has
become a global cultural icon of France and one of the most recognizable structures in the world.
```

**2. Modify `rag_pipeline.py`:**

In `rag_pipeline.py`, replace this:

```python
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
```

with this:

```python
# --- 1. Document Loading ---
# We load our document from a text file.
with open("rag_pipeline/data.txt", "r") as f:
    document = f.read()
```

## Exercise 2: Different Embedding Model

To use a different embedding model, you just need to change the `model_name` in the `HuggingFaceEmbeddings` constructor.

For example, to use the `distilbert-base-nli-stsb-mean-tokens` model, you would change this:

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

to this:

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
```

Remember that using a different model will require downloading its weights, which might take some time and disk space.

## Exercise 3: Add More Questions

You can add more questions by simply calling the `answer_question` function with your new questions. For example:

```python
# ... (rest of the code)

print("--- Querying the RAG Pipeline ---")
answer_question("Where is the Eiffel Tower located?")
answer_question("Who designed the Eiffel Tower?")
answer_question("When was the Eiffel Tower built?")
answer_question("Why was the Eiffel Tower built?")

# --- New Questions ---
answer_question("What is the Eiffel Tower made of?")
answer_question("How long did it take to build the Eiffel Tower?")
answer_question("What was the initial reaction to the Eiffel Tower?")
```

## Exercise 4: Explore the Retriever

You can control the number of documents the retriever fetches by using the `search_kwargs` parameter in the `as_retriever` method.

For example, to retrieve the top 3 most similar documents, you would change this:

```python
retriever = db.as_retriever()
```

to this:

```python
retriever = db.as_retriever(search_kwargs={"k": 3})
```

By experimenting with the value of `k`, you can see how retrieving more or fewer documents affects the context provided to the language model and, consequently, the quality of the final answer.
