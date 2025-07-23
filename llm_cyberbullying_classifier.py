import os
import sys
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Ensure GROQ_API_KEY is set
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    print("Error: Please set your GROQ_API_KEY environment variable.")
    sys.exit(1)

# Choose the best available model (as of 2024, 'mixtral-8x7b-32768' is a top choice)
MODEL_NAME = "gemma2-9b-it"

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are an expert in online safety. "
        "Given the following message, classify whether it is cyberbullying or not. "
        "Respond with the following format:\n"
        "Cyberbullying: <Yes/No>\n"
        "Confidence: <0-100>%\n"
        "Explanation: <short explanation>\n"
        "\nMessage: {text}"
    )
)

# Initialize the LLM
llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.0,
    groq_api_key=GROQ_API_KEY,
)

def classify_message(message: str):
    system_prompt = (
        "You are an expert in online safety. "
        "Given the following message, classify whether it is cyberbullying or not. "
        "Respond with the following format:\n"
        "Cyberbullying: <Yes/No>\n"
        "Confidence: <0-100>%\n"
        "Explanation: <short explanation>\n"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message)
    ]
    response = llm.invoke(messages)
    return response.content

def main():
    print("Cyberbullying Classifier (Groq + LangChain)")
    print("Type your message and press Enter. Type 'exit' to quit.\n")
    while True:
        user_input = input("Input: ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue
        print("\nClassifying...\n")
        result = classify_message(user_input)
        print(result)
        print("\n---\n")

if __name__ == "__main__":
    main() 