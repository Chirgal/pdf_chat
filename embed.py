import os
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
import io
import streamlit as st

os.environ["ASTRA_DB_APPLICATION_TOKEN"] = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
os.environ["ASTRA_DB_API_ENDPOINT"] = st.secrets["ASTRA_DB_API_ENDPOINT"]

@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Instantiate a HuggingFaceEmbeddings object with the specified model name.

    Args:
        model_name (str, optional): The name of the pre-trained model to be
                                    used for embedding. Defaults to 'all-MiniLM-L6-v2'.

    Returns:
        HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings configured
                               with the specified model.
    """
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embedding = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    return embedding


def load_vector_store(collection_name, model_name="all-MiniLM-L6-v2"):
    """
    Initialize AstraDB Vector Store

    Args:
        collection_name (str): Collection name for document store.
        model_name (str, optional): Hugginface embedding model name. Defaults to "all-MiniLM-L6-v2".

    Returns:
        AstraDBVectorStore: Handle to vector store.
    """
    embedding = load_embedding_model(model_name)
    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name=collection_name,
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    )
    
    return vstore

def load_pdfs_to_vector_store(docs, vectore_store):
    documents = []
    for doc in docs:
        loader = PyPDFLoader(doc)
        pages = loader.load_and_split()
        documents.extend(pages)
    vectore_store.add_documents(documents)
    
    
def load_pdfs_to_vector_store2(docs, vectore_store):
    documents = []
    for doc in docs:
        reader = PdfReader(doc)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            text = ""
            text += reader.pages[page_num].extract_text()
            metadata = {'source' : doc.name, 'page_num' : page_num}
            document = Document(page_content=text, metadata=metadata)
            documents.append(document)
    vectore_store.add_documents(documents)
    
def get_top_context(vectore_store, ques):
    results = vectore_store.similarity_search(ques, k=1)
    print(results)
    return results[0].page_content
    
    
if __name__ == '__main__':
    
    ques = "What is stirred-tank reactor ?"
    docs = ["D:\\Downloads\\pdf\\lebo109.pdf"]
    
    vstore = load_vector_store('test')
    # load_pdfs_to_vector_store(docs, vstore)
    
    results = vstore.similarity_search(ques, k=3)
    print(results)
    # for res in results:
    #     print(f"* {res.page_content} [{res.metadata}]")
