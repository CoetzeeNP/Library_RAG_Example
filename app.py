import streamlit as st
from openai import OpenAI

# ============================================
# 🔹 CONFIG
# ============================================
st.set_page_config(page_title="RAG Chatbot (Responses API)", layout="centered")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# IMPORTANT: Your vector store ID (from OpenAI dashboard)
VECTOR_STORE_ID = st.secrets["VECTOR_STORE_ID"]

# ============================================
# 🔹 SESSION STATE
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================
# 🔹 SIDEBAR
# ============================================
with st.sidebar:
    st.title("⚙️ Settings")

    rag_enabled = st.toggle("Enable RAG (File Search)", value=True)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ============================================
# 🔹 MAIN UI
# ============================================
st.title("🤖 RAG Chatbot (Responses API)")
st.caption("Powered by File Search (Vector Store)")

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================
# 🔹 USER INPUT
# ============================================
if prompt := st.chat_input("Ask something..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            # ====================================
            # 🔹 BUILD INPUT
            # ====================================
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            # ====================================
            # 🔹 RAG MODE (File Search)
            # ====================================
            if rag_enabled:
                stream = client.responses.stream(
                    model="gpt-4.1",  # or gpt-4o / gpt-5
                    input=messages,
                    tools=[{
                        "type": "file_search",
                        "vector_store_ids": [VECTOR_STORE_ID],
                    }],
                )
            else:
                stream = client.responses.stream(
                    model="gpt-4.1",
                    input=messages,
                )

            # ====================================
            # 🔹 STREAM RESPONSE
            # ====================================
            for event in stream:
                if event.type == "response.output_text.delta":
                    full_response += event.delta
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            placeholder.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
