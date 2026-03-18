import streamlit as st
from openai import OpenAI

# ============================================
# 🔹 CONFIGURATION
# ============================================
st.set_page_config(
    page_title="RAG Chatbot",
    layout="centered"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
VECTOR_STORE_ID = st.secrets["VECTOR_STORE_ID"]

# ============================================
# 🔹 SESSION STATE INIT
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True

# ============================================
# 🔹 SIDEBAR
# ============================================
with st.sidebar:
    st.image("icdf.png")
    st.title("⚙️ Settings")

    rag_toggle = st.toggle("Enable RAG (File Search)", value=st.session_state.rag_enabled)

    # 🔥 Reset chat if toggle changes (IMPORTANT)
    if rag_toggle != st.session_state.rag_enabled:
        st.session_state.rag_enabled = rag_toggle
        st.session_state.messages = []
        st.rerun()

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ============================================
# 🔹 MAIN UI
# ============================================
st.image("combined_logo.jpg")
st.title("RAG Chatbot")
st.caption("Demonstrating Retrieval-Augmented Generation")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================
# 🔹 USER INPUT
# ============================================
if prompt := st.chat_input("Ask something..."):
    st.markdown("---")
    st.markdown("### Mode")
    st.markdown(
        "📚 **RAG Enabled**" if st.session_state.rag_enabled
        else "💬 **Standard Chat (No RAG)**")
    
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
            # ====================================
            # 🔹 BUILD MESSAGE HISTORY
            # ====================================
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            # ====================================
            # 🔹 DYNAMIC SYSTEM PROMPT
            # ====================================
            if st.session_state.rag_enabled:
                system_prompt = (
                    "Answer using ONLY the knowledge base. "
                    "If the answer is not found, say you don't know."
                )
            else:
                system_prompt = (
                    "You are a helpful AI assistant. "
                    "Answer normally without using external documents."
                )

            messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })

            # ====================================
            # 🔹 BUILD REQUEST
            # ====================================
            request_params = {
                "model": "gpt-4.1",
                "input": messages,
            }

            # ✅ Only include RAG when enabled
            if st.session_state.rag_enabled:
                request_params["tools"] = [{
                    "type": "file_search",
                    "vector_store_ids": [VECTOR_STORE_ID],
                }]

            # ====================================
            # 🔹 STREAM RESPONSE (CORRECT)
            # ====================================
            with client.responses.stream(**request_params) as stream:

                for event in stream:
                    if event.type == "response.output_text.delta":
                        full_response += event.delta
                        placeholder.markdown(full_response + "▌")

                # Get final response safely
                final_response = stream.get_final_response()

            # ====================================
            # 🔹 OPTIONAL: CITATIONS
            # ====================================
            if st.session_state.rag_enabled:
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

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
