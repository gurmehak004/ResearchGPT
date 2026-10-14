import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_groq_client = None


def _get_groq_client() -> Groq:
    """Lazy initialize Groq client."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set in .env.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def get_llm_response(query: str, context: str) -> tuple[str, float]:
    """
    Generate grounded response using high-speed Qwen 27B / Llama 3 on Groq Cloud.
    Target inference latency: ~0.5s - 1.2s.

    Args:
        query: The user's question
        context: Retrieved document chunks as text

    Returns:
        Tuple of (answer_text, confidence_score)
    """
    system_prompt = f"""You are an expert Academic Research Assistant. Your ONLY job is to answer questions strictly using the provided context from the uploaded documents.

STRICT INSTRUCTIONS:
1. Ground your answer completely in the context below.
2. If the user asks for an abstract, summary, or general overview, synthesize the response directly from the document header and introduction chunks provided.
3. Multi-Paper Analysis: If the context contains multiple papers, compare their methods, conclusions, and benchmarks explicitly.
4. Citations: When stating specific findings or data, name the source paper (e.g., "[paper_name.pdf]").
5. If the context does not contain enough information to answer, state clearly: "The uploaded documents do not contain sufficient information to answer this question."
6. Do NOT invent facts or rely on external training knowledge.

CONTEXT FROM DOCUMENTS:
{context}

OUTPUT FORMAT:
Respond strictly in valid JSON with:
{{
  "answer": "Your detailed, formatted answer in markdown.",
  "confidence": 0.95
}}
Confidence range: 0.0 to 1.0."""

    try:
        client = _get_groq_client()
        # qwen/qwen3.8-27b delivers ~400 tokens/sec on Groq LPUs
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            model="qwen/qwen3.8-27b",
            response_format={"type": "json_object"},
            temperature=0.2,
            max_completion_tokens=1500,
        )

        response_content = chat_completion.choices[0].message.content or "{}"
        result = json.loads(response_content)
        answer = result.get("answer", "I could not generate an answer from the documents.")
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))

    except json.JSONDecodeError:
        answer = response_content if "response_content" in locals() and response_content else "The model returned an unparseable response."
        confidence = 0.4
    except Exception as e:
        print(f"[ERROR] Groq LLM Error: {e}")
        answer = f"Error communicating with LLM: {str(e)}."
        confidence = 0.0

    return answer, confidence


def check_hallucination(answer: str, context: str) -> bool:
    """
    Evaluates if the answer claims facts not in the context.
    """
    eval_prompt = f"""You are an objective academic evaluator.
Check if the generated answer invents definitive numbers, claims, or citations that are unsupported or directly contradicted by the context.

CONTEXT:
{context}

ANSWER:
{answer}

Respond strictly in JSON:
{{"hallucinated": false}}
"""
    try:
        client = _get_groq_client()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": eval_prompt}],
            model="qwen/qwen3.8-27b",
            response_format={"type": "json_object"},
            temperature=0.0,
            max_completion_tokens=60,
        )
        response_content = chat_completion.choices[0].message.content or "{}"
        result = json.loads(response_content)
        return bool(result.get("hallucinated", False))
    except Exception as e:
        print(f"[ERROR] Verification Error: {e}")
        return False