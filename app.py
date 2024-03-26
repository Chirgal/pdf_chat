import streamlit as st
import time
import requests
from embed import load_pdfs_to_vector_store2, load_vector_store, get_top_context, fetch_doc_names, create_sqlite_table
import io, math

MODEL1 = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
MODEL2 = 'mistralai/Mistral-7B-Instruct-v0.2'
MODEL3 = 'HuggingFaceH4/starchat2-15b-v0.1'
MODEL = MODEL2
COLLECTION_NAME = 'PDF_for_chat'

API_URL = F"https://api-inference.huggingface.co/models/{MODEL}"
headers = {"Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}"}
create_sqlite_table()

@st.cache_resource
def get_vectore_store():
    return load_vector_store(COLLECTION_NAME)

def query(payload):
    payload = {
	      "inputs": payload,
           'parameters': { 'max_new_tokens':500,'return_full_text': False}
              }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def stream(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)



st.set_page_config(
    page_title="PDF ChatBot - Demo",
    page_icon=":robot:"
)

st.header("PDF ChatBot")
st.write("Upload your PDF documents and ask me anything about them. I will try to answer.")

vstore= get_vectore_store()
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.write("Upload and Process PDF document(s) to chat.")
    uploaded_files = st.file_uploader("Choose a file", type=['pdf'], accept_multiple_files=True)
    # print(f"uploaded_files = {uploaded_files}")
    
    if st.button("Process"):
        with st.spinner("Processing"):
            if len(uploaded_files) > 0:
                load_pdfs_to_vector_store2(uploaded_files, vstore)
                st.info("Done")
            else:
                st.warning("No file(s) to process !")
    
    doc_names = fetch_doc_names()         
    options = st.multiselect(
    '*OR* you can choose from uploaded PDF documents.',
    doc_names,
    )

    
# Accept user input
if prompt := st.chat_input("Ask me anything about your pdf document"):
    context, score = get_top_context(vstore, prompt, list(set(options)))
    if round(score, 1) <= 0.6:
        prompt_inst = f"[INST]Reply you are not able to answer.[/INST]"
    else:
        prompt_inst = f"[INST]{prompt}\nUse only below context to generate a valid short answer:\n\n{context}[/INST]"
    print(prompt_inst)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    fetch = True
    with st.chat_message("assistant"):
        while fetch:
            try:
                with st.spinner('Typing...'):
                    output = query(prompt_inst)
                    # print(output)
                    output_text = output[0]["generated_text"]
                    fetch = False
            # print(output)
            except KeyError:
                fetch = True
        st.write_stream(stream(output_text))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": output_text})

#print(st.session_state.messages)