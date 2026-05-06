import sys
import os

sys.path.append(os.path.abspath("."))

from app.main import ask_question

# Test dataset
test_data = [
    {
        "question": "What tools are mentioned in the resume?",
        "expected_keywords": [
            "Power BI",
            "Tableau",
            "SQL",
            "Python"
        ]
    },
    {
        "question": "What role did the candidate work as?",
        "expected_keywords": [
            "Data Analyst"
        ]
    }
]

total = 0
correct = 0

print("\n--- RAG Evaluation ---\n")

for item in test_data:

    question = item["question"]
    expected_keywords = item["expected_keywords"]

    response = ask_question(question)

    answer = response["answer"]

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")

    matched = False

    for keyword in expected_keywords:
        if keyword.lower() in answer.lower():
            matched = True
            break

    if matched:
        print("✅ PASS")
        correct += 1
    else:
        print("❌ FAIL")

    total += 1

score = correct / total

print("\n-------------------")
print(f"Final Score: {score:.2f}")
print("-------------------")