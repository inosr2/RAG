import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
from tenacity import retry, wait_fixed, stop_after_attempt
import os
import google.generativeai as genai

# --- Configuration and Setup ---
try:
    GOOGLE_API_KEY = "AIzaSyCfLE97x4S3KzMOYQ8MLnKZzrPaRGfxNhE"
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    st.error("Google API Key not found. Please set it in your Streamlit secrets.")
    st.stop()

st.set_page_config(page_title="Ask your PDF")
st.header("Ask questions about your PDF 📚")

# --- Session State Initialization ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# --- File Upload and Processing ---
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None and not st.session_state.pdf_processed:
    with st.spinner("Processing PDF... This may take a moment."):
        try:
            pdf_reader = PdfReader(pdf)
            text = "".join([page.extract_text() for page in pdf_reader.pages])

            if not text.strip():
                st.warning("Could not extract text from the PDF. It might be scanned or image-based.")
                st.session_state.pdf_processed = False
                st.stop()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            if not chunks:
                st.error("No text chunks could be created from the PDF.")
                st.session_state.pdf_processed = False
                st.stop()

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                task_type="RETRIEVAL_DOCUMENT"
            )

            @retry(wait=wait_fixed(5), stop=stop_after_attempt(5), reraise=True)
            def create_vector_store_with_retry():
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

            create_vector_store_with_retry()
            st.session_state.pdf_processed = True
            st.success("PDF processed successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error processing PDF: {e}. Please try again or use a different PDF.")
            st.session_state.pdf_processed = False

# --- Question Handling ---
col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input("Ask a question:")
with col2:
    use_ai_only = st.checkbox("Ask by AI")

if question:
    if use_ai_only:
        with st.spinner("Thinking with AI..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.5,
                )
                response = llm.invoke(question)
                # st.write("### AI Answer:")
                # st.info(response.text())
                st.markdown(f"""
                <div style="
                    background-color: #e9f7ef;
                    padding: 1.2rem;
                    border-radius: 12px;
                    border: 1px solid #b5eacb;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                    font-size: 1.05rem;
                    line-height: 1.6;
                    color: #1a1a1a;
                    margin-top: 1rem;
                ">
                    <strong>🤖 AI Answer:</strong><br><br>
                    {response.text()}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"AI failed to answer: {e}")
    elif st.session_state.pdf_processed and st.session_state.vector_store is not None:
        with st.spinner("Finding answer..."):
            try:
                docs = st.session_state.vector_store.similarity_search(question, k=4)

                if not docs:
                    st.warning("Could not find relevant information in the PDF for your question.")
                    st.stop()

                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.3,
                )

                chain = load_qa_chain(llm, chain_type="stuff")

                @retry(wait=wait_fixed(10), stop=stop_after_attempt(3), reraise=True)
                def invoke_qa_chain():
                    return chain.invoke({"input_documents": docs, "question": question})["output_text"]

                response = invoke_qa_chain()
                # st.write("### Answer:")
                # st.info(response)
                # st.markdown(f"{response}", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="
                    background-color: #f0f4ff;
                    padding: 1.2rem;
                    border-radius: 12px;
                    border: 1px solid #d3e3fd;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                    font-size: 1.05rem;
                    line-height: 1.6;
                    color: #1a1a1a;
                    margin-top: 1rem;
                ">
                    <strong>🧠 Answer:</strong><br><br>
                    {response}
                </div>
                """, unsafe_allow_html=True)


            except Exception as e:
                st.error(f"Error generating answer: {e}")
                st.write("You might be hitting API rate limits or there might be an issue with the model response.")
                st.write("If the problem persists, check your Google Cloud Console for API usage and errors.")
    else:
        st.warning("Upload a PDF first to use PDF-based answering.")
elif pdf is None:
    st.info("Upload a PDF file to start asking questions.")
