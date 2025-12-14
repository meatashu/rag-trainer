
# RAG Pipeline Exercises

Here are a few exercises to help you practice and extend your knowledge of RAG pipelines.

## Exercise 1: Different Data Source

Modify the `rag_pipeline.py` script to load its document from a text file named `data.txt`.

1.  Create a file named `data.txt` in the `rag_pipeline` directory.
2.  Add some text to this file. You can use the provided example text or write your own.
3.  Modify the `rag_pipeline.py` script to read the content of `data.txt` instead of using the hardcoded string.

## Exercise 2: Different Embedding Model

The choice of embedding model can have a significant impact on the performance of your RAG pipeline.

1.  Go to the [Hugging Face Model Hub](https://huggingface.co/models) and find a different sentence transformer model.
2.  Modify the `rag_pipeline.py` script to use this new model.
3.  Observe if there is any change in the quality of the answers.

## Exercise 3: Add More Questions

Add more questions to the `rag_pipeline.py` script to test the pipeline's knowledge.

1.  Come up with a few more questions about the Eiffel Tower.
2.  Add them to the `answer_question` calls at the end of the script.
3.  See how the pipeline responds to these new questions.

## Exercise 4: Explore the Retriever

The retriever plays a crucial role in the RAG pipeline.

1.  Experiment with the retriever's configuration. For example, you can try to change the number of documents it retrieves. (Hint: Look into the `as_retriever` method's `search_kwargs` parameter).
2.  See how changing the retriever's settings affects the final answers.
