import streamlit as st
import asyncio
import os

import ingest
import search_agent
import logs


# ============================================================================
# Configuration
# ============================================================================

API_KEY = #Add your key here
os.environ['MISTRAL_API_KEY'] = API_KEY


# ============================================================================
# Initialization
# ============================================================================

@st.cache_resource
def init_agent():
    repo_owner = "DataTalksClub"
    repo_name = "faq"

    def filter(doc):
        return "data-engineering" in doc["filename"]

    st.write("🔄 Indexing repo...")
    index = ingest.index_data(repo_owner, repo_name, filter=filter)
    agent = search_agent.init_agent(index, repo_owner, repo_name)
    return agent


agent = init_agent()


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="AI FAQ Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI FAQ Assistant")
st.caption("Ask me anything about the DataTalksClub/faq repository")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================================
# Chat input
# ============================================================================

if prompt := st.chat_input("Ask your question..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message (non-streaming due to Mistral compatibility issues)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(agent.run(user_prompt=prompt))
            answer = response.output
            st.markdown(answer)

    # Save response to history + logs
    st.session_state.messages.append({"role": "assistant", "content": answer})
    logs.log_interaction_to_file(agent, response.new_messages())
