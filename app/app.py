"""Hansard RAG — Streamlit UI.

Run: streamlit run app.py
"""

import streamlit as st

from hansard_rag.db import init_db, log_conversation, log_feedback
from hansard_rag.rag import rag

st.set_page_config(page_title="Ask Parliament", page_icon="🏛️", layout="wide")

init_db()

st.title("🏛️ Ask Parliament")
st.caption(
    "Ask what's been said in UK parliamentary debates — answers grounded in Hansard, "
    "with speaker and party attribution. Try: *What have MPs said about NHS dentistry?* "
    "or *Summarise the Doncaster Royal Infirmary debate* or *What has Bob Blackman been raising?*"
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: question, result, conversation_id, feedback_given

question = st.chat_input("Ask about UK parliamentary debates...")

if question:
    with st.spinner("Searching Hansard..."):
        result = rag(question)
    conversation_id = log_conversation(question, result)
    st.session_state.history.append(
        {"question": question, "result": result, "conversation_id": conversation_id, "feedback_given": False}
    )

for i, turn in enumerate(st.session_state.history):
    result = turn["result"]
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(result["answer"])
        st.caption(f"route: {result['route']} · {result['latency_s']}s · {len(result['sources'])} sources")

        with st.expander("Sources"):
            for r in result["sources"]:
                speaker = r["speaker"] + (f" ({r['party']})" if r.get("party") else "")
                st.markdown(
                    f"**{r['debate_title']}** ({r['sitting_date']}) — {speaker}  \n"
                    f"{r['text'][:300]}...  \n"
                    f"[Read in Hansard]({r['hansard_url']})"
                )

        if not turn["feedback_given"]:
            col_up, col_down, _ = st.columns([1, 1, 8])
            if col_up.button("👍", key=f"up_{i}"):
                log_feedback(turn["conversation_id"], True)
                turn["feedback_given"] = True
                st.rerun()
            if col_down.button("👎", key=f"down_{i}"):
                log_feedback(turn["conversation_id"], False)
                turn["feedback_given"] = True
                st.rerun()
        else:
            st.caption("Thanks for the feedback!")
