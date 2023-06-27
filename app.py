import os
from config import Openai_api_key
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
os.environ['OPENAI_API_KEY'] = Openai_api_key

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

st.title('🦜🔗 Xpay Level II GPT Summarizer')
st.success('This App allows you to summarize the financial health of a company after uploading its Annual document')

# Create a file uploader in Streamlit
uploaded_file = st.file_uploader("Upload The Annual Financial document (PDF)", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("uploaded_document.pdf", "wb") as file:
        file.write(uploaded_file.read())

    # Load the uploaded PDF document using PyPDFLoader
    loader = PyPDFLoader("uploaded_document.pdf")

    # Split pages from the PDF
    pages = loader.load_and_split()

    # Load documents into the vector database (ChromaDB)
    store = Chroma.from_documents(pages, embeddings, collection_name='uploaded_document')

    # Create vectorstore info object
    vectorstore_info = VectorStoreInfo(
        name="uploaded_document",
        description="Uploaded financial document as a PDF",
        vectorstore=store
    )

    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    # Create a text input box for the user
    prompt = st.text_input('Input your prompt here')

    # If the user hits enter
    if prompt:
        # Then pass the prompt to the LLM
        response = agent_executor.run(prompt)
        # ...and write it out to the screen
        st.write(response)

        # With a streamlit expander  
        with st.expander('Document Similarity Search'):
            # Find the relevant pages
            search = store.similarity_search_with_score(prompt) 
            # Write out the first 
            st.write(search[0][0].page_content)

    # Delete the temporary file
    os.remove("uploaded_document.pdf")
