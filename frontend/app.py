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
        st.log(f"Using backend URL from Streamlit secrets: {backend_url}")
    else:
        # Fallback for local development
        st.log("Loading backend URL from local config.toml")
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

user_query = st.text_area("Ask Howie a question about making a cup of coffee:") 

if user_query:
    query = {"query": user_query}

    # Use a spinner for better user experience while waiting for the API
    with st.spinner("Howie is thinking..."):
        try:
            # Send the query to the correct backend endpoint
            response = requests.post(ASK_ENDPOINT, json=query)

            if response.status_code == 200:
                data = response.json()
                st.write("**Answer:**")
                st.markdown(data.get("answer", "No answer found.")) # Use markdown for better formatting
                
                st.write("**Sources:**")
                # Create a set to handle potential duplicate sources cleanly
                unique_sources = set()
                for source in data.get("sources", []):
                    if 'file_name' in source:
                        # Clean up the file path for display
                        display_name = os.path.basename(source['file_name'])
                        unique_sources.add(display_name)
                    elif 'source' in source:
                        unique_sources.add(source['source'].replace('_', ' ').title())
                
                for source_name in sorted(list(unique_sources)):
                    st.write(f"- {source_name}")
            else:
                # Display a more user-friendly error
                st.error(f"Error from backend: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the backend at {ASK_ENDPOINT}. Please ensure it is running. Error: {e}")


