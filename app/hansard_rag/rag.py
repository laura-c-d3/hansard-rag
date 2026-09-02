"""RAG layer: prompts, the two tools, and the router.

Extracted from rag.ipynb. V2 (attributed) prompt is the default per the LLM eval.
"""

import json
import os
import time

from groq import Groq

from .search import find_debate, get_debate_chunks, hybrid_search

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "llama-3.1-8b-instant")

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def llm(prompt, model=GROQ_MODEL, temperature=0.0, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            usage = response.usage
            return response.choices[0].message.content.strip(), {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2 ** (attempt + 2))
            else:
                raise
    raise RuntimeError("rate limited: gave up after retries")


def build_context(results):
    blocks = []
    for r in results:
        speaker = r["speaker"] + (f" ({r['party']})" if r.get("party") else "")
        blocks.append(
            f"Debate: {r['debate_title']} ({r['sitting_date']})\n"
            f"Speaker: {speaker}\n"
            f"Text: {r['text']}"
        )
    return "\n\n---\n\n".join(blocks)


ANSWER_PROMPT = """You are an assistant answering questions about UK parliamentary debates.
Answer the QUESTION using only the CONTEXT below.

Rules:
- Attribute every claim to the speaker who made it, with their party where available,
  e.g. 'Bob Blackman (Con) argued that...'
- Where speakers disagree, present the different positions side by side
- If the context does not contain enough information, say so plainly
- End with a 'Sources' line listing the debates and dates used

QUESTION: {question}

CONTEXT:
{context}"""

SUMMARISE_PROMPT = """You are summarising a UK parliamentary debate.

Summarise the debate below:
- Lead with what the debate was about and its outcome if stated
- Cover the main points raised, attributed to speaker (party)
- Note points of disagreement between speakers
- Keep it under 300 words

DEBATE: {title} ({date})

TRANSCRIPT:
{transcript}"""

ROUTER_PROMPT = """Classify this question about UK parliamentary debates into exactly one route:

- SEARCH: asking what was said about a topic across parliament, by any/many speakers
- DEBATE_SUMMARY: asking to summarise or explain one specific named debate
- SPEAKER: asking what one specific named MP has said (about anything or a topic)

Question: {question}

Respond with ONLY a JSON object: {{"route": "...", "debate_title": "...", "speaker_name": "...", "topic": "..."}}
Use null for fields that do not apply."""


def _parse_json(raw):
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def route_question(question):
    raw, usage = llm(ROUTER_PROMPT.format(question=question), model=ROUTER_MODEL)
    try:
        return _parse_json(raw), usage
    except json.JSONDecodeError:
        return {"route": "SEARCH", "topic": question}, usage


def answer_from_search(question, k=5, filters=None):
    results = hybrid_search(question, k=k, filters=filters)
    prompt = ANSWER_PROMPT.format(question=question, context=build_context(results))
    answer, usage = llm(prompt)
    return answer, results, usage


def summarise_debate(title_query):
    found = find_debate(title_query)
    if not found:
        return f"No debate found matching '{title_query}'", [], {"prompt_tokens": 0, "completion_tokens": 0}
    ext_id, title = found
    chunks = get_debate_chunks(ext_id)
    transcript = build_context(chunks)
    if len(transcript) > 60000:
        transcript = transcript[:60000] + "\n\n[transcript truncated]"
    prompt = SUMMARISE_PROMPT.format(title=title, date=chunks[0]["sitting_date"], transcript=transcript)
    answer, usage = llm(prompt)
    return answer, chunks, usage


def rag(question):
    """Route, execute, and return everything the app/monitoring needs."""
    start = time.time()
    decision, router_usage = route_question(question)
    route = decision.get("route", "SEARCH")

    if route == "DEBATE_SUMMARY" and decision.get("debate_title"):
        answer, sources, usage = summarise_debate(decision["debate_title"])
    elif route == "SPEAKER" and decision.get("speaker_name"):
        query = decision.get("topic") or question
        results = hybrid_search(query, k=8)
        named = [r for r in results if decision["speaker_name"].lower() in (r["speaker"] or "").lower()]
        sources = named or results
        prompt = ANSWER_PROMPT.format(question=question, context=build_context(sources))
        answer, usage = llm(prompt)
    else:
        route = "SEARCH"
        answer, sources, usage = answer_from_search(question)

    return {
        "answer": answer,
        "route": route,
        "sources": sources,
        "latency_s": round(time.time() - start, 2),
        "prompt_tokens": usage["prompt_tokens"] + router_usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"] + router_usage["completion_tokens"],
    }
