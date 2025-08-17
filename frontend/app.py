import streamlit as st
import requests
import tomllib  # Use 'tomli' for Python < 3.11 if needed
import os


# --- 1. Centralized Configuration Function ---
def get_ask_endpoint_url() -> str:
    """
    Determines the correct backend API endpoint URL based on the environment.
    Reads from Streamlit Secrets in production or a local config.toml for development.
    """
    # Check if running on Streamlit Community Cloud (where secrets are set)
    if "BACKEND_URL" in st.secrets:
        backend_url = st.secrets["BACKEND_URL"]
        print(f"Using backend URL from Streamlit secrets: {backend_url}")
    else:
        # Fallback for local development
        print("Loading backend URL from local config.toml")
        try:
            # Assumes your streamlit app is run from the 'howie' project root
            with open("config.toml", "rb") as f:
                config = tomllib.load(f)
            backend_url = config.get("api", {}).get("backend_url", "http://127.0.0.1:8000")
        except FileNotFoundError:
            backend_url = "http://127.0.0.1:8000"
    
    # Append the specific endpoint path
    return f"{backend_url}/ask"

# --- 2. Main Application ---
st.title("Howie - Your AI Assistant")
st.write("This is the frontend for Howie, your AI assistant powered by LlamaIndex and Vertex AI.")

# Get the endpoint URL ONCE at the start
ASK_ENDPOINT = get_ask_endpoint_url()
print(f"Using Ask endpoint: {ASK_ENDPOINT}")

# def send_message():
#     st.write(f"Sent message: {st.session_state.chat_message}")

# st.chat_input("Type your message...", on_submit=send_message, key="chat_message")

# user_query = st.text_area("Ask Howie a question about making a cup of coffee:") 

# if user_query:
#     query = {"query": user_query}

#     # Use a spinner for better user experience while waiting for the API
#     with st.spinner("Howie is thinking..."):
#         try:
#             # Send the query to the correct backend endpoint
#             response = requests.post(ASK_ENDPOINT, json=query)

#             if response.status_code == 200:
#                 data = response.json()
#                 st.write("**Answer:**")
#                 st.markdown(data.get("answer", "No answer found.")) # Use markdown for better formatting
                
#                 st.write("**Sources:**")
#                 # Create a set to handle potential duplicate sources cleanly
#                 unique_sources = set()
#                 for source in data.get("sources", []):
#                     if 'file_name' in source:
#                         # Clean up the file path for display
#                         display_name = os.path.basename(source['file_name'])
#                         unique_sources.add(display_name)
#                     elif 'source' in source:
#                         unique_sources.add(source['source'].replace('_', ' ').title())
                
#                 for source_name in sorted(list(unique_sources)):
#                     st.write(f"- {source_name}")
#             else:
#                 # Display a more user-friendly error
#                 st.error(f"Error from backend: {response.status_code} - {response.text}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Could not connect to the backend at {ASK_ENDPOINT}. Please ensure it is running. Error: {e}")


# st.title("Howie - Your AI Assistant")
# st.write("This is the frontend for Howie, your AI assistant...")

# --- 1. Initialize chat history in session_state ---
# This is a best practice for chat apps
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. Display past messages ---
# Loop through the saved messages and display them in the chat container
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Optionally display sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                     st.write(source)

# --- 3. Use st.chat_input for new user input ---
# This widget will appear at the bottom of the screen
if prompt := st.chat_input("Ask Howie a question..."):
    # Add user message to the chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 4. Get and display the assistant's response ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # For a streaming effect
        full_response_text = ""
        
        with st.spinner("Howie is thinking..."):
            try:
                # Send the query to the backend API
                response = requests.post(ASK_ENDPOINT, json={"query": prompt})

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "I'm sorry, I encountered an issue.")
                    sources_data = data.get("sources", [])
                    
                    # You can add a simple "typing" effect here if you want
                    # Or just display the final answer
                    message_placeholder.markdown(answer)
                    
                    # Format sources for display
                    sources_formatted = []
                    unique_sources = set()
                    for source in sources_data:
                        if 'file_name' in source:
                            display_name = os.path.basename(source['file_name'])
                            unique_sources.add(display_name)
                        elif 'source' in source:
                            unique_sources.add(source['source'].replace('_', ' ').title())
                    
                    sources_formatted = sorted(list(unique_sources))

                    if sources_formatted:
                        with st.expander("View Sources"):
                            for source_name in sources_formatted:
                                st.write(f"- {source_name}")
                    
                    # Add the full response to the session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources_formatted # Save the cleaned sources
                    })

                else:
                    message_placeholder.error(f"Error from backend: {response.status_code}")
                    st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, there was an error."})
            
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the backend. Error: {e}")
