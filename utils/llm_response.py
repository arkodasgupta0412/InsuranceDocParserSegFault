import os
from dotenv import load_dotenv
from google import genai
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

INSURANCE_PROMPT_TEMPLATE = """
You are an intelligent insurance parser.

Answer the following insurance-related question professionally and clearly.
Answer in a single line in a short and concise form taken from relevant clauses. However do not mention these clauses in the answe

Relevant clauses:
{context}

Question:
{query}

Respond in markdown using a professional tone.
"""

genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_structured_output(query, clauses):
    context = "\n\n".join([
        f"[Clause from {doc.metadata.get('source', 'unknown')}]:\n{doc.page_content}"
        for doc in clauses
    ])
    prompt = INSURANCE_PROMPT_TEMPLATE.format(query=query, context=context)
    
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
    except Exception as e:
        return f"Error: Failed to generate response for query '{query}': {str(e)}"

    return response.text.strip()

def generate_structured_answers(questions, vectorstore):
    def process(q):
        clauses = get_relevant_clauses(q, vectorstore)
        return generate_structured_output(q, clauses)
    
    from utils.retriever import get_relevant_clauses

    with ThreadPoolExecutor() as executor:
        answers = list(executor.map(process, questions))

    return answers