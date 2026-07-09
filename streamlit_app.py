import streamlit as st

from app.fireworks_client import FireworksConfig, FireworksClient
from app.graph_router import LangGraphRouter


st.set_page_config(page_title="TokenPilot", page_icon="⚡", layout="centered")
st.title("TokenPilot")
st.caption("Adaptive Fireworks model routing for cost-efficient answers")

prompt = st.text_area("Enter a task", height=160, placeholder="Ask a question or describe a task...")

with st.sidebar:
    st.subheader("Configuration")
    st.write("The app uses FIREWORKS_API_KEY from Streamlit Secrets.")
    if st.button("Clear conversation"):
        st.session_state.pop("router", None)
        st.rerun()


def get_router() -> LangGraphRouter:
    if "router" not in st.session_state:
        import os

        for name in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
            if name in st.secrets:
                os.environ[name] = str(st.secrets[name])
        # Keep deployment simple: the public demo only needs the API key secret.
        os.environ.setdefault(
            "ALLOWED_MODELS",
            "accounts/fireworks/models/llama-v3p1-8b-instruct,"
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
        )
        config = FireworksConfig.from_environment()
        client = FireworksClient(config)
        st.session_state.router = LangGraphRouter(client)
    return st.session_state.router


if st.button("Route and answer", type="primary", disabled=not prompt.strip()):
    try:
        with st.spinner("Selecting the most efficient model..."):
            result = get_router().run(prompt.strip())
        st.success("Answer generated")
        answer = result["answer"]
        st.markdown(answer.answer)
        left, right = st.columns(2)
        left.metric("Selected tier", answer.tier.value)
        right.metric("Answer tokens", answer.token_count)
        with st.expander("Routing details"):
            st.json({
                "tier": answer.tier.value,
                "confidence": answer.confidence,
                "cache_hit": answer.cache_hit,
                "attempts": result.get("attempts", []),
            })
    except Exception as exc:
        st.error("The demo could not run. Check FIREWORKS_API_KEY and ALLOWED_MODELS in Streamlit Secrets.")
        st.exception(exc)
