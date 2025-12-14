import ollama

ollama_base_url="http://localhost:11434"
client = ollama.Client(host=ollama_base_url)
def get_ollama_response(model_name: str, prompt: str) -> str:
    """Get a response from a local Ollama model."""
    response = client.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def show_ollama_models():
    """Print the available Ollama models from the local instance."""
    models = client.list()["models"]
    print("Available Ollama Models:")
    # ollama_models = [m["name"] for m in client.list()["models"]]
    for model in models:
        # model='ministral-3:latest' modified_at=datetime.datetime(2025, 12, 12, 1, 36, 6, 125954, tzinfo=TzInfo(-18000)) digest='a5e54193fd347063e4f9cdcf37fde6907a37c6c91100ccf6ec3aebe1fb8259e0' size=6022236223 details=ModelDetails(parent_model='', format='gguf', family='mistral3', families=['mistral3'], parameter_size='8.9B', quantization_level='Q4_K_M')
        # model='mistral:latest' modified_at=datetime.datetime(2025, 11, 23, 23, 25, 59, 848490, tzinfo=TzInfo(-18000)) digest='6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf' size=4372824384 details=ModelDetails(parent_model='', format='gguf', family='llama', families=['llama'], parameter_size='7.2B', quantization_level='Q4_K_M')
        # model='gemma3:1b' modified_at=datetime.datetime(2025, 10, 5, 15, 37, 8, 418211, tzinfo=TzInfo(-14400)) digest='8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc' size=815319791 details=ModelDetails(parent_model='', format='gguf', family='gemma3', families=['gemma3'], parameter_size='999.89M', quantization_level='Q4_K_M')
        print(f"- Model Name: {model['model']}, Size: {model['size']}, Modified At: {model['modified_at']}, details: {model['details']['parameter_size']}, {model['details']['quantization_level']}, Family: {model['details']['family']}")

if __name__ == "__main__":
    show_ollama_models()