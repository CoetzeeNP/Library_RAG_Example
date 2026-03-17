import streamlit as st
from openai import OpenAI

# ============================================
# 🔹 CONFIGURATION
# ============================================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
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
st.caption("Ask questions about your knowledge base")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================
# 🔹 USER INPUT
# ============================================
if prompt := st.chat_input("Ask something..."):

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # ============================================
    # 🔹 ASSISTANT RESPONSE
    # ============================================
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            # Build message history
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            # Create streaming request
            with client.responses.stream(
                model="gpt-4.1",  # or gpt-4o / gpt-5
                input=messages,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [VECTOR_STORE_ID],
                }] if rag_enabled else None,
            ) as stream:

                citations = set()

                for event in stream:

                    # ✅ Stream text output
                    if event.type == "response.output_text.delta":
                        full_response += event.delta
                        placeholder.markdown(full_response + "▌")

                    # ✅ Capture RAG citations
                    elif event.type == "response.file_search_call.completed":
                        for result in event.output:
                            citations.add(result["file_id"])

                # Add citations at the end
                if citations:
                    full_response += "\n\n---\n📄 **Sources:**\n"
                    for c in citations:
                        full_response += f"- `{c}`\n"

                # Final render
                placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            placeholder.markdown(full_response)

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
