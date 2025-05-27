import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from tenacity import retry, wait_fixed, stop_after_attempt
import os
import google.generativeai as genai
from googletrans import Translator
import pdfplumber
from io import BytesIO

# --- Configuration ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    st.error("Google API Key not found. Please set it in your Streamlit secrets.")
    st.stop()

translator = Translator()

def detect_language(text):
    try:
        return translator.detect(text).lang 
    except:
        return "en"

def translate_text(text, target_lang):
    if not text.strip():
        return text
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text



st.set_page_config(page_title="DocQuery AI | Ask your PDF")
st.header("Ask questions about your PDF 📚")


# --- Session State ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "document_lang" not in st.session_state:
    st.session_state.document_lang = "en"

# --- Upload and Extract with pdfplumber Only ---
uploaded_file = st.file_uploader("Upload your PDF or Image", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None and not st.session_state.pdf_processed:
    with st.spinner("Processing document..."):
        try:
            text = ""
            if uploaded_file.type == "application/pdf":
                from pdf2image import convert_from_bytes
                import pytesseract

                # Try extracting with pdfplumber
                with pdfplumber.open(BytesIO(uploaded_file.read())) as plumber_pdf:
                    text = "\n".join([
                        page.extract_text() for page in plumber_pdf.pages if page.extract_text()
                    ])

                if not text.strip():
                    st.warning("No text found. Trying OCR...")
                    images = convert_from_bytes(uploaded_file.getvalue())
                    text = "\n".join([pytesseract.image_to_string(img) for img in images])

            elif uploaded_file.type.startswith("image/"):
                import pytesseract
                from PIL import Image
                image = Image.open(uploaded_file)
                text = pytesseract.image_to_string(image)

            if not text.strip():
                st.error("Could not extract any text from the uploaded file.")
                st.session_state.pdf_processed = False
                st.stop()


            document_lang = detect_language(text)
            st.session_state.document_lang = document_lang

            # st.text("complete text extracted from the PDF and language detected:")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            if not chunks:
                st.error("No text chunks could be created from the PDF.")
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
            st.error(f"Error processing PDF: {e}")
            st.session_state.pdf_processed = False

# --- Ask Questions ---
col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input("Ask a question:")
with col2:
    use_ai_only = st.checkbox("Ask by AI")

if question:
    user_lang = detect_language(question)
    doc_lang = st.session_state.get("document_lang", "en")

    translated_question = (
        translate_text(question, doc_lang)
        if user_lang != doc_lang else question
    )

    if use_ai_only:
        with st.spinner("Thinking with AI..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.5,
                )
                response = llm.invoke(translated_question)
                answer_text = response.text()

                if user_lang != doc_lang:
                    answer_text = translate_text(answer_text, user_lang)

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
                    {answer_text}
                </div>
                """, unsafe_allow_html=True)
               
            except Exception as e:
                st.error(f"AI failed to answer: {e}")

    elif st.session_state.pdf_processed and st.session_state.vector_store is not None:
        with st.spinner("Finding answer..."):
            try:
                docs = st.session_state.vector_store.similarity_search(translated_question, k=4)
                if not docs:
                    st.warning("Could not find relevant information in the PDF.")
                    st.stop()

                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.3,
                )
                chain = load_qa_chain(llm, chain_type="stuff")

                @retry(wait=wait_fixed(10), stop=stop_after_attempt(3), reraise=True)
                def invoke_qa_chain():
                    return chain.invoke({
                        "input_documents": docs,
                        "question": translated_question
                    })["output_text"]

                response = invoke_qa_chain()
                if user_lang != doc_lang:
                    response = translate_text(response, user_lang)

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
    else:
        st.warning("Upload a PDF first to use PDF-based answering.")
elif uploaded_file is None:
    st.info("Upload a PDF file to start asking questions.")
