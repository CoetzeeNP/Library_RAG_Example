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

            # Optional system instruction (helps RAG quality)
            messages.insert(0, {
                "role": "system",
                "content": "Answer using the knowledge base. If unsure, say you don't know."
            })

            # ✅ Always use a list for tools
            tools = [{
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
            }] if rag_enabled else []

            # ✅ Correct streaming usage
            with client.responses.stream(
                model="gpt-4.1",
                input=messages,
                tools=tools,
            ) as stream:

                for event in stream:
                    if event.type == "response.output_text.delta":
                        full_response += event.delta
                        placeholder.markdown(full_response + "▌")

                # ✅ Get final response safely
                final_response = stream.get_final_response()

            # ====================================
            # 🔹 OPTIONAL: Extract citations safely
            # ====================================
            try:
                annotations = final_response.output[0].content[0].annotations

                if annotations:
                    full_response += "\n\n---\n📄 **Sources:**\n"
                    for ann in annotations:
                        if hasattr(ann, "file_citation"):
                            file_id = ann.file_citation.file_id
                            full_response += f"- `{file_id}`\n"
            except Exception:
                pass

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
