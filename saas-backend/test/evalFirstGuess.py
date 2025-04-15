from openai import OpenAI
import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

EVALUATION_PROMPT_TEMPLATE = """
You are an expert tutor evaluating a chatbot response. Read the provided scenario carefully and evaluate the chatbot based on these criteria using a 1-5 scale. Provide each score with a short justification.

Criteria:
1. Response correctness (accuracy of chatbot’s answer)
2. Quality of feedback (effectiveness of chatbot’s feedback on student's guess. Use the provided 'First Guess Analysis' for clarity; mark N/A if no guess provided.)
3. Relevance (chatbot’s response relevance)
4. Educational effectiveness (clarity and educational value)
5. Engagement potential (motivation to continue interaction)

Scenario:
Question: {question_text}
Student's First Guess: {student_guess}
Chatbot Response: {chatbot_response}
First Guess Analysis: {firstguess_analysis}

Your Evaluation:
- Response correctness: 
- Quality of feedback: 
- Relevance: 
- Educational effectiveness: 
- Engagement potential: 
"""

# Function to evaluate a single response using the LLM
def evaluate_response(question_text, student_guess, chatbot_response, firstguess_analysis):
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question_text=question_text,
        student_guess=student_guess,
        chatbot_response=chatbot_response,
        firstguess_analysis=firstguess_analysis
    )
    completion = client.chat.completions.create(
        model='gpt-4-turbo',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500
    )
    return completion.choices[0].message.content.strip()

# Load your JSON data
with open('responses.json', 'r') as file:
    data = json.load(file)

# Evaluate each response
results = []
for entry in tqdm(data):
    evaluation = evaluate_response(
        entry['question_text'],
        entry['student_guess'],
        entry['chatbot_response'],
        entry['firstguess_analysis']
    )
    results.append({
        "question_id": entry["question_id"],
        "response_type": entry["response_type"],
        "evaluation": evaluation
    })

# Save evaluations to CSV
df_results = pd.DataFrame(results)
df_results.to_csv('chatbot_evaluation_results.csv', index=False)