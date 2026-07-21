import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load HF_TOKEN from your .env file
load_dotenv()

client = InferenceClient(token=os.environ["HF_TOKEN"])

def ask_agent(prompt, model="meta-llama/Llama-3.1-8B-Instruct"):
    """Send a prompt to the Hugging Face agent and return its response."""
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("Chat with your agent! Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        reply = ask_agent(user_input)
        print(f"Agent: {reply}\n")
        