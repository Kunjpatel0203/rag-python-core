from dotenv import load_dotenv
load_dotenv()

from agents.intent_decomposer import decompose_intent

questions = [
    "What is the DA for April and what does the policy say about it?",
    "What is the opening stock of Laptop?",
    "Explain the compliance policy for DA",
    "Give me salary details and policy explanation"
]

for q in questions:
    print("\nQUESTION:", q)
    print("OUTPUT:", decompose_intent(q))
