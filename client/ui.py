import streamlit as st
import requests
import time

# --- CONFIGURATION ---
# Ensure your Master Node is running on this port
MASTER_URL = "http://localhost:8080/query"

st.set_page_config(
    page_title="Distributed RAG Interface",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .stDeployButton {display:none;}
    .reportview-container { background: #f0f2f6; }
    .worker-tag {
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 10px;
        background-color: #e1e4e8;
        color: #0366d6;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Distributed RAG System")
st.caption("Connected to Master Node | Load Balanced Architecture")

# User Input Section
with st.container():
    query = st.chat_input("Ask a question about your documents...")

    if query:
        # 1. Display User Message
        with st.chat_message("user"):
            st.markdown(query)

        # 2. Call Master Node
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.markdown("*(Wait)* Master is routing to an idle worker...")
            
            payload = {
                "user_id": "Renad_Local_Dev",
                "query": query,
                "user_sent_at": int(time.time())
            }
            
            try:
                # Headers for your API key
                headers = {"x-api-key": "dev-key-1", "Content-Type": "application/json"}
                
                start_time = time.time()
                response = requests.post(MASTER_URL, json=payload, headers=headers, timeout=300)
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    worker_id = data.get("worker_id", "Unknown Node")
                    latency = round(end_time - start_time, 2)

                    # Clear status and show response
                    status_placeholder.empty()
                    st.markdown(answer)
                    
                    # 3. Transparent Metadata (The "Standout" Footer)
                    st.markdown(f"---")
                    st.markdown(
                        f"<span class='worker-tag'>Processed by: {worker_id}</span> "
                        f"<span class='worker-tag'>E2E Latency: {latency}s</span>", 
                        unsafe_allow_html=True
                    )
                    
                    # Expandable section for RAG context (if your worker returns it)
                    if "context" in data:
                        with st.expander("View Retrieved Document Chunks"):
                            st.write(data["context"])
                
                else:
                    status_placeholder.error(f"Master Error ({response.status_code}): {response.text}")

            except Exception as e:
                status_placeholder.error(f"Failed to connect to Master: {str(e)}")

# Sidebar for System Info
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("Master: Connected")
    st.divider()
    st.markdown("""
    **Architecture:**
    - Master (Port 8080)
    - Workers (Port 8081, 8082)
    - LLM: Qwen 2.5 (A6000)
    """)
    if st.button("Clear Chat History"):
        st.rerun()