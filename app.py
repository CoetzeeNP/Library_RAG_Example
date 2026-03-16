import streamlit as st
from openai import OpenAI
import time

# --- INITIALIZATION ---
st.set_page_config(page_title="RAG Toggle Chatbot", layout="centered")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Assistant ID from your OpenAI Platform (where RAG is configured)
ASSISTANT_ID = "asst_your_assistant_id_here"

# Initialize session state for chat history and thread
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    # Create a persistent thread for the Assistant
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("Settings")
    # This is the button/toggle you requested
    rag_enabled = st.toggle("Enable RAG (Assistant Mode)", value=False)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Chatbot")
st.info(f"Current Mode: {'**RAG Enabled** (using Assistant)' if rag_enabled else '**Standard GPT**'}")

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        if rag_enabled:
            # --- RAG MODE: Use Assistants API ---
            # 1. Add message to the thread
            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=prompt
            )
            # 2. Run the assistant
            run = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=ASSISTANT_ID
            )
            
            # 3. Wait for completion (Polling)
            while run.status != "completed":
                time.sleep(0.5)
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread_id, run_id=run.id
                )
            
            # 4. Get the latest message
            messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
            full_response = messages.data[0].content[0].text.value
        
        else:
            # --- STANDARD MODE: Use Chat Completions ---
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
