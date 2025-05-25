import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter # Changed from CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from PyPDF2 import PdfReader # Keep PyPDF2 but note alternatives
from tenacity import retry, wait_fixed, stop_after_attempt, stop_never # Added stop_never for indefinite retries on connection issues
import os
import google.generativeai as genai # Import for direct API key configuration if needed by the library

# --- Configuration and Setup ---

# Use Streamlit's built-in secrets management for API key security
# Access the API key from st.secrets.
# For local development, create a .streamlit/secrets.toml file in your app's root directory
# [secrets]
# GOOGLE_API_KEY = "your_google_api_key_here"
# On Streamlit Community Cloud, set this in the app's secrets settings.
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    # It's good practice to set the API key in the environment for Langchain/Google GenAI
    # as some internal components might pick it up directly from os.environ
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    # Also configure the google-generativeai library directly
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    st.error("Google API Key not found. Please set it in your Streamlit secrets.")
    st.stop() # Stop the app if API key is missing

# Configure Streamlit page
st.set_page_config(page_title="Ask your PDF")
st.header("Ask questions about your PDF 📚")

# --- Session State Initialization ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False # Track if PDF has been processed

# --- File Upload and Processing ---
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None and not st.session_state.pdf_processed:
    with st.spinner("Processing PDF... This may take a moment."):
        try:
            # Read PDF
            pdf_reader = PdfReader(pdf)
            text = "".join([page.extract_text() for page in pdf_reader.pages])

            if not text.strip():
                st.warning("Could not extract text from the PDF. It might be scanned or image-based. Consider using a different library like `pdfminer.six` or `PDFplumber` for better text extraction.")
                st.session_state.pdf_processed = False
                st.stop()

            # Split text into chunks using RecursiveCharacterTextSplitter
            # This splitter is better for semantic coherence in RAG applications
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Increased chunk size for potentially better context
                chunk_overlap=200, # Increased overlap for better context flow
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            if not chunks:
                st.error("No text chunks could be created from the PDF. The PDF might be empty or problematic.")
                st.session_state.pdf_processed = False
                st.stop()

            # Create embeddings using Google's model
            # Added task_type for optimized embeddings in retrieval tasks
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="RETRIEVAL_DOCUMENT")

            # Retry mechanism for vector store creation
            # Adjusted retry strategy for robustness
            @retry(wait=wait_fixed(5), stop=stop_after_attempt(5), reraise=True)
            def create_vector_store_with_retry():
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

            create_vector_store_with_retry()
            st.session_state.pdf_processed = True
            st.success("PDF processed successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error processing PDF: {e}. Please try again or use a different PDF.")
            st.session_state.pdf_processed = False # Reset flag on error

# --- Question Handling ---
if st.session_state.pdf_processed and st.session_state.vector_store is not None:
    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Finding answer..."):
            try:
                # Search for similar chunks in the vector store
                # Number of retrieved documents can be adjusted (e.g., k=5)
                docs = st.session_state.vector_store.similarity_search(question, k=4)

                if not docs:
                    st.warning("Could not find relevant information in the PDF for your question.")
                    st.stop()

                # Initialize Google's LLM
                # No 'api_key' parameter needed if GOOGLE_API_KEY is in os.environ
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash", # Use 'model' parameter
                    temperature=0.3,
                )

                # Load the QA chain
                chain = load_qa_chain(llm, chain_type="stuff")

                # Invoke the chain
                # Added retry for the LLM invocation as well
                @retry(wait=wait_fixed(10), stop=stop_after_attempt(3), reraise=True)
                def invoke_qa_chain():
                    return chain.invoke({"input_documents": docs, "question": question})["output_text"]

                response = invoke_qa_chain()

                st.write("### Answer:")
                st.info(response) # Use st.info for a distinct answer box

            except Exception as e:
                st.error(f"Error generating answer: {e}. Please try again.")
                st.write("You might be hitting API rate limits or there might be an issue with the model response.")
                st.write("If the problem persists, check your Google Cloud Console for API usage and errors.")

elif pdf is None:
    st.info("Upload a PDF file to start asking questions.")
elif st.session_state.pdf_processed and st.session_state.vector_store is None:
    st.warning("PDF was processed, but the vector store is not available. Please try uploading again.")