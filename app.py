import streamlit as st
from openai import OpenAI
import time

# ============================================
# 🔹 CONFIGURATION
# ============================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
VECTOR_STORE_ID = st.secrets["VECTOR_STORE_ID"]
ASSISTANT_ID = st.secrets.get("ASSISTANT_ID") # You'll need an Assistant ID linked to your VS

# ============================================
# 🔹 SESSION STATE
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    # Create a persistent thread for the conversation
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

# ============================================
# 🔹 MAIN UI
# ============================================
st.title("🤖 RAG Chatbot with Chunk Inspection")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chunks" in msg:
            with st.expander("🔍 Viewed Source Chunks"):
                for chunk in msg["chunks"]:
                    st.caption(f"File: {chunk['file_id']} | Score: {chunk['score']}")
                    st.write(chunk['text'])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Add Message to Thread
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=prompt
        )

        # 2. Run the Assistant
        with st.status("Searching knowledge base & generating...") as status:
            run = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=ASSISTANT_ID,
                # Ensure file_search is enabled for this run
                tools=[{"type": "file_search"}] 
            )

            # 3. Wait for completion (Polling)
            while run.status in ["queued", "in_progress"]:
                time.sleep(0.5)
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread_id, 
                    run_id=run.id
                )

            # 4. FETCH THE CHUNKS (The magic part)
            retrieved_chunks = []
            steps = client.beta.threads.runs.steps.list(
                thread_id=st.session_state.thread_id, 
                run_id=run.id
            )
            
            for step in steps.data:
                if step.step_details.type == "tool_calls":
                    for tool in step.step_details.tool_calls:
                        if hasattr(tool, "file_search"):
                            # This retrieves the specific snippets found
                            results = tool.file_search.get("results", [])
                            for res in results:
                                retrieved_chunks.append({
                                    "file_id": res.get("file_id"),
                                    "score": round(res.get("score", 0), 3),
                                    "text": res.get("content", [{}])[0].get("text", "No text found")
                                })
            
            status.update(label="Response ready!", state="complete")

        # 5. Get the final text response
        messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
        full_response = messages.data[0].content[0].text.value
        
        # Display response
        st.markdown(full_response)
        
        # Display chunks in an expander
        if retrieved_chunks:
            with st.expander("🔍 Viewed Source Chunks"):
                for chunk in retrieved_chunks:
                    st.caption(f"File: {chunk['file_id']} | Relevance: {chunk['score']}")
                    st.write(chunk['text'])

        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "chunks": retrieved_chunks
        })
