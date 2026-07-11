"""
GeoIPS — Streamlit Chat Interface.

A premium web UI for the Plane Geometry Intelligent Problem Solver.
Supports:
- Natural language geometry queries (English & Vietnamese)
- AlphaGeometry-style solving via /geo/solve (with auxiliary constructions)
- Live-streaming proof explanations
- Proof history sidebar
"""

import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
DOMAIN = "geometry"

EXAMPLE_QUERIES = [
    "Cho tam giác ABC cân tại A. Chứng minh góc B bằng góc C.",
    "Given right triangle ABC right-angled at A, prove BC²=AB²+AC².",
    "If Congruent(AB,CD) and Congruent(CD,EF), prove Congruent(AB,EF).",
    "If Parallel(a,b) and Parallel(b,c), prove Parallel(a,c).",
    "Given Triangle(A,B,C) and Congruent(AB,BC) and Congruent(BC,AC), prove Equal(Angle(BAC),60).",
]


def check_backend_health() -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def solve_query(query: str, use_aux_agent: bool = True) -> dict:
    """POST query to solver. Uses /geo/solve (AlphaGeometry) or /api/solve."""
    endpoint = "/geo/solve" if use_aux_agent else "/api/solve"
    try:
        payload = (
            {"query": query, "max_construction_iterations": 2}
            if use_aux_agent
            else {"query": query, "domain": DOMAIN}
        )
        resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "❌ Cannot reach backend. Please ensure the GeoIPS server is running."}
    except requests.exceptions.Timeout:
        return {"error": "⏱️ Request timed out. The solver may be working on a complex proof."}
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"error": f"⚠️ Backend error: {detail}"}
    except Exception as e:
        return {"error": f"❌ Unexpected error: {str(e)}"}


def explain_proof_stream(query: str, execution_trace: list, goal_reached: bool, aux_constructions: list):
    """Stream proof explanation from backend."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/explain/stream",
            json={
                "query": query,
                "domain": DOMAIN,
                "execution_trace": execution_trace,
                "goal_reached": goal_reached,
                "auxiliary_constructions": aux_constructions,
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")()
        for raw_chunk in resp.iter_content(chunk_size=1024):
            if raw_chunk:
                chunk = decoder.decode(raw_chunk)
                for ch in chunk:
                    yield ch
                    time.sleep(0.002)
        final = decoder.decode(b"", final=True)
        for ch in final:
            yield ch
            time.sleep(0.002)
    except Exception as e:
        yield f"\n\n*Stream error: {str(e)}*"


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoIPS — Plane Geometry IPS",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Premium Design
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a1a;
    color: #e0e0f0;
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0b1f 0%, #121230 60%, #0b0b1f 100%);
    border-right: 1px solid #1e1e4a;
  }
  section[data-testid="stSidebar"] .stMarkdown h1,
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3,
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] .stMarkdown li {
    color: #c8c8ff !important;
  }

  /* Status badges */
  .status-badge {
    padding: 5px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    display: inline-block; margin: 3px 0;
    letter-spacing: 0.5px;
  }
  .status-online  { background: linear-gradient(135deg, #00c853, #00e676); color: #003300; }
  .status-offline { background: linear-gradient(135deg, #ff1744, #ff5252); color: #fff; }

  /* Proof result badges */
  .goal-badge {
    padding: 10px 22px; border-radius: 14px;
    font-size: 1.05rem; font-weight: 700;
    display: inline-block; margin: 10px 0;
    letter-spacing: 0.3px;
  }
  .goal-reached { background: linear-gradient(135deg, #00c853, #69f0ae); color: #002200; }
  .goal-missed  { background: linear-gradient(135deg, #ff6f00, #ffab40); color: #3e2700; }

  /* Proof path display */
  .proof-path {
    background: linear-gradient(135deg, #0d1226, #141840);
    border: 1px solid #1e2a6a;
    border-left: 4px solid #5c6bc0;
    border-radius: 10px; padding: 14px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem; color: #90caf9;
    letter-spacing: 0.8px; margin: 8px 0;
  }

  /* Fact chips */
  .fact-chip {
    display: inline-block;
    background: linear-gradient(135deg, #4a4fbf, #6c63ff);
    color: #fff; padding: 4px 13px; border-radius: 18px;
    font-size: 0.82rem; font-weight: 500; margin: 3px 4px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Auxiliary construction chip — distinct amber color */
  .aux-chip {
    display: inline-block;
    background: linear-gradient(135deg, #e65100, #ff8f00);
    color: #fff; padding: 4px 13px; border-radius: 18px;
    font-size: 0.82rem; font-weight: 500; margin: 3px 4px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Main header */
  .main-header { text-align: center; padding: 1.5rem 0 0.5rem 0; }
  .main-header h1 {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #6c63ff 0%, #48c9b0 60%, #f4d03f 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
  }
  .main-header p { color: #8888b0; font-size: 0.95rem; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Chat area */
  .stChatMessage { border-radius: 12px; }

  /* Sidebar proof history items */
  .history-item {
    background: rgba(100, 100, 255, 0.08);
    border: 1px solid rgba(100, 100, 255, 0.2);
    border-radius: 8px; padding: 8px 12px; margin: 6px 0;
    font-size: 0.82rem; color: #b0b0d8; cursor: pointer;
  }
  .history-item:hover { border-color: #6c63ff; color: #fff; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 0.2rem 0;">
      <span style="font-size:2.5rem;">📐</span>
      <h2 style="font-size:1.4rem; font-weight:800; background: linear-gradient(135deg,#6c63ff,#48c9b0);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0.2rem 0;">GeoIPS</h2>
      <p style="font-size:0.78rem; color:#8888b0; margin:0;">Plane Geometry IPS</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Backend status
    st.markdown("### ⚙️ System Status")
    backend_url_input = st.text_input("API URL", value=BACKEND_URL, label_visibility="collapsed")
    if backend_url_input != BACKEND_URL:
        BACKEND_URL = backend_url_input

    health = check_backend_health()
    if health:
        neo4j_ok = health.get("neo4j_connected", False)
        qdrant_mode = health.get("qdrant_mode", "?")
        qdrant_ep = health.get("qdrant_endpoint", "?")

        st.markdown(
            f'<span class="status-badge status-online">● Online</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"- Neo4j: {'✅' if neo4j_ok else '⚠️ offline'}")
        st.markdown(f"- Qdrant: `{qdrant_mode}` → `{qdrant_ep}`")
    else:
        st.markdown(
            '<span class="status-badge status-offline">● Offline</span>',
            unsafe_allow_html=True,
        )
        st.caption("Start with `make run-server` or `uv run uvicorn api.main:app`")

    st.markdown("---")

    # AlphaGeometry mode toggle
    st.markdown("### 🧠 Solver Mode")
    use_aux = st.toggle(
        "AlphaGeometry Mode",
        value=True,
        help="When ON: uses /geo/solve with auxiliary construction loop. When OFF: uses standard /api/solve.",
    )
    if use_aux:
        st.caption("🔬 Auxiliary construction agent active — can add geometric objects when stuck.")
    else:
        st.caption("⚡ Standard forward-chaining mode.")

    st.markdown("---")

    # Example queries
    st.markdown("### 💡 Example Queries")
    for ex in EXAMPLE_QUERIES:
        if st.button(f"📎 {ex[:55]}...", key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state["prefilled_query"] = ex

    st.markdown("---")

    # Proof history
    st.markdown("### 📖 Proof History")
    if "proof_history" not in st.session_state:
        st.session_state.proof_history = []

    if st.session_state.proof_history:
        for item in reversed(st.session_state.proof_history[-10:]):
            icon = "✅" if item["proved"] else "❌"
            st.markdown(
                f'<div class="history-item">{icon} {item["query"][:60]}...</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No proofs yet. Ask a geometry question!")

    st.markdown("---")
    st.caption("Powered by **Neo4j** · **Qdrant** · **LangChain**")


# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
  <h1>📐 GeoIPS Chat</h1>
  <p>Neuro-Symbolic Plane Geometry Solver — inspired by AlphaGeometry</p>
  <p style="font-size:0.82rem; color:#6666aa;">Ask in natural language (English or Vietnamese) or use formal predicates.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "proof_history" not in st.session_state:
    st.session_state.proof_history = []

# ---------------------------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if "error" in msg:
                st.error(msg["error"])
            else:
                result = msg.get("result", {})
                explanation = msg.get("explanation", "")
                goal_reached = result.get("goal_reached", False)

                # Goal badge
                if goal_reached:
                    st.markdown('<div class="goal-badge goal-reached">✅ Proof Successful!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="goal-badge goal-missed">⚠️ Goal Not Reached</div>', unsafe_allow_html=True)

                # Auxiliary constructions alert
                aux_constructions = result.get("auxiliary_constructions", [])
                if aux_constructions:
                    st.info(f"🔧 **Auxiliary Constructions Added:** {len(aux_constructions)} new object(s) helped bridge the proof gap.")
                    chips = " ".join([f'<span class="aux-chip">🔧 {f}</span>' for f in aux_constructions])
                    st.markdown(chips, unsafe_allow_html=True)

                # GraphRAG Info expander
                with st.expander("🔍 GraphRAG Processing Details", expanded=False):
                    st.markdown("**Qdrant-Mapped Initial Facts:**")
                    mapped_facts = result.get("mapped_initial_facts", [])
                    if mapped_facts:
                        chips = " ".join([f'<span class="fact-chip">{f}</span>' for f in mapped_facts])
                        st.markdown(chips, unsafe_allow_html=True)
                    else:
                        st.caption("No facts mapped.")

                    st.markdown(f"**Mapped Goal:** `{result.get('mapped_goal', 'N/A')}`")

                    trace = result.get("execution_trace", [])
                    st.markdown("**Execution Trace:**")
                    if trace:
                        for step in trace:
                            st.code(
                                f"Rule: {step.get('rule_id','?')} → {step.get('fired_rule_repr','?')}\n"
                                f"New facts: {step.get('new_facts',[])}",
                                language="text",
                            )
                    else:
                        st.caption("No rules fired.")

                    known = result.get("known_facts", [])
                    if known:
                        st.markdown("**Final Known Facts:**")
                        st.code(", ".join(known), language="text")

                # Proof path
                rule_ids = result.get("applied_rule_ids", [])
                if rule_ids:
                    trace_str = " → ".join(rule_ids)
                    st.markdown(
                        f'<div class="proof-path">📋 {trace_str}</div>',
                        unsafe_allow_html=True,
                    )

                # LLM Explanation
                st.markdown("---")
                if explanation:
                    st.markdown(explanation)
                else:
                    st.info("No explanation generated.")

# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------
prefilled = st.session_state.pop("prefilled_query", None)
user_input = st.chat_input(
    placeholder="e.g. Cho tam giác ABC cân tại A. Chứng minh góc B bằng góc C.",
    key="chat_input",
)
if prefilled and not user_input:
    user_input = prefilled

if user_input:
    st.session_state.messages.append({
        "role": "user", "content": user_input, "avatar": "👤"
    })
    st.rerun()

# ---------------------------------------------------------------------------
# Assistant Turn
# ---------------------------------------------------------------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_msg = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant", avatar="📐"):
        with st.spinner("🔄 Running Neuro-Symbolic inference engine..."):
            result = solve_query(user_msg, use_aux_agent=use_aux)

        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append({
                "role": "assistant", "avatar": "📐", "error": result["error"]
            })
            st.rerun()
        else:
            goal_reached = result.get("goal_reached", False)
            aux_constructions = result.get("auxiliary_constructions", [])

            # Goal badge
            if goal_reached:
                st.markdown('<div class="goal-badge goal-reached">✅ Proof Successful!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="goal-badge goal-missed">⚠️ Goal Not Reached</div>', unsafe_allow_html=True)

            # Auxiliary constructions
            if aux_constructions:
                st.info(f"🔧 **AlphaGeometry Mode:** Added {len(aux_constructions)} auxiliary construction(s) to bridge the proof gap.")
                chips = " ".join([f'<span class="aux-chip">🔧 {f}</span>' for f in aux_constructions])
                st.markdown(chips, unsafe_allow_html=True)

            # GraphRAG detail
            with st.expander("🔍 GraphRAG Processing Details", expanded=False):
                st.markdown("**Qdrant-Mapped Initial Facts:**")
                mapped_facts = result.get("mapped_initial_facts", [])
                if mapped_facts:
                    chips = " ".join([f'<span class="fact-chip">{f}</span>' for f in mapped_facts])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("No facts mapped.")

                st.markdown(f"**Mapped Goal:** `{result.get('mapped_goal', 'N/A')}`")

                trace = result.get("execution_trace", [])
                st.markdown("**Execution Trace:**")
                if trace:
                    for step in trace:
                        st.code(
                            f"Rule: {step.get('rule_id','?')} → {step.get('fired_rule_repr','?')}\n"
                            f"New facts: {step.get('new_facts',[])}",
                            language="text",
                        )
                else:
                    st.caption("No rules fired.")

                known = result.get("known_facts", [])
                if known:
                    st.markdown("**Final Known Facts:**")
                    st.code(", ".join(known), language="text")

            # Proof path
            rule_ids = result.get("applied_rule_ids", [])
            if rule_ids:
                trace_str = " → ".join(rule_ids)
                st.markdown(
                    f'<div class="proof-path">📋 {trace_str}</div>',
                    unsafe_allow_html=True,
                )

            # Live streaming explanation
            st.markdown("---")
            explanation_str = ""
            trace_data = result.get("execution_trace", [])
            if trace_data or aux_constructions:
                explanation_str = st.write_stream(
                    explain_proof_stream(user_msg, trace_data, goal_reached, aux_constructions)
                )
            else:
                explanation_str = (
                    "No inference steps were generated. "
                    "The solver found no applicable rules. "
                    "Try rephrasing your query or running `make embed-knowledge` to populate the KB."
                )
                st.info(explanation_str)

            # Save to history
            st.session_state.proof_history.append({
                "query": user_msg,
                "proved": goal_reached,
                "rule_count": len(rule_ids),
            })

            st.session_state.messages.append({
                "role": "assistant",
                "avatar": "📐",
                "result": result,
                "explanation": explanation_str,
            })
            st.rerun()
