import os
import streamlit as st
import tempfile
import PyPDF2
import shutil
from io import BytesIO
from crewai import Agent, Crew, Task
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource
def load_local_llm():
    pipe = pipeline(
        "text-generation",
        model="microsoft/phi-2",   # small, fast, free
        max_new_tokens=512,
        temperature=0.3,
        device=-1  # CPU; use 0 if you have GPU
    )
    return HuggingFacePipeline(pipeline=pipe)

st.set_page_config(
    page_title="Research Paper Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'paper_summary' not in st.session_state:
    st.session_state.paper_summary = None


def extract_text_from_pdf(uploaded_file):

    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:

            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
        return text

    except Exception as e:

        st.error(f"Error extracting text from PDF: {str(e)}")

        return None
    

def generate_paper_summary(text):
    try:
        #llm = OpenAI(temperature=0.3, max_tokens=1000)  
        #llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3, max_tokens=1000)
        llm = load_local_llm()
        summary_prompt = (
            "### Instruction:\n"
            "Provide a structured summary of the following research paper with sections:\n"
            "Title, Abstract, Methodology, Key Findings, Contributions, Limitations, Future Work.\n\n"
            "### Research Paper:\n"
            f"{text[:6000]}\n\n"
            "### Summary:\n"
        )

        summary = llm.invoke(summary_prompt)
        #summary = llm.invoke(summary_prompt).content

        return summary

    except Exception as e:

        st.error(f"Error generating summary: {str(e)}")

        return None
    
def cleanup_chroma_db():
    try:
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
    except Exception as e:
        st.warning(f"Could not clean up ChromaDB directory: {str(e)}")

def process_document(text):
    try:
        cleanup_chroma_db()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )

        docs = splitter.create_documents([text])

        try:
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                ),
                collection_name="research_papers"
            )

            return vectorstore

        except Exception as chroma_error:
            st.warning(f"ChromaDB failed, trying FAISS: {str(chroma_error)}")
            vectorstore = FAISS.from_documents(
                documents=docs,
                embedding = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            )

            return vectorstore            

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        cleanup_chroma_db()

        return None
    
def retrieval_action(question, vectorstore):

    results = vectorstore.similarity_search(question, k=5)

    return "\n\n".join([f"Passage {i+1}: {doc.page_content}" for i, doc in enumerate(results)])

def generation_action(inputs):

    if isinstance(inputs, dict):
        question = inputs.get("user", "")
        context = inputs.get("Retriever", "")
    else:
        question = "User question not found in inputs"
        context = str(inputs)
    prompt = (
        "### Instruction:\n"
        "Answer the question using ONLY the provided passages. "
        "Use numbered citations [1], [2], etc. "
        "If the answer is not contained in the passages, say so.\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Passages:\n"
        f"{context}\n\n"
        "### Answer:\n"
    )
    llm = load_local_llm()
    return llm.invoke(prompt)
    
    #llm = OpenAI(model = "gpt-3.5-turbo-instruct" , temperature=0.2, max_tokens=1500)
    #return llm.invoke(prompt)

    #llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2, max_tokens=1500)
    #return llm.invoke(prompt).content

def create_crew(vectorstore, status_callback=None):
    def retrieval_wrapper(question):
        if status_callback:
            status_callback("Searching for relevant passages in the research paper...")

        result = retrieval_action(question, vectorstore)

        if status_callback:
            status_callback("Found relevant passages for analysis")

        return result

    def generation_wrapper(inputs):

        if status_callback:
            status_callback("Analyzing retrieved passages and generating comprehensive answer...")

        result = generation_action(inputs)

        if status_callback:
            status_callback("Generated detailed answer with citations")

        return result

    retriever = Agent(
        role="Retriever",
        goal="Retrieve relevant passages from the research paper",
        backstory="You are an expert at finding relevant information in research papers",
        action=retrieval_wrapper,
        verbose=True
    )

    generator = Agent(
        role="Generator", 
        goal="Generate comprehensive answers based on retrieved passages",
        backstory="You are an expert research analyst who provides detailed, citation-based answers",
        action=generation_wrapper,
        verbose=True
    )

    retrieval_task = Task(
        description="Retrieve relevant passages from the research paper for the user's question",
        agent=retriever,
        expected_output="Relevant passages from the research paper that can answer the user's question"
    )

    generation_task = Task(
        description="Generate a comprehensive answer with citations based on the retrieved passages from the research paper",
        agent=generator,
        expected_output="A detailed answer with numbered citations based on the research paper content",
        context=[retrieval_task]
    )    

    crew = Crew(
        agents=[retriever, generator],
        tasks=[retrieval_task, generation_task],
        verbose=True
    )

    return crew


def main():
    st.title("Research Paper Analyst")
    st.markdown("Upload a research paper and ask questions about it using AI-powered analysis.")    
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a research paper in PDF format"
        )        
        if uploaded_file is not None:
            st.info(f"File uploaded: {uploaded_file.name}")        
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document and generating summary..."):
                    text = extract_text_from_pdf(uploaded_file)    
                    if text:
                        summary = generate_paper_summary(text)        
                        vectorstore = None  
                        if summary:
                            st.session_state.paper_summary = summary
                            vectorstore = process_document(text)        
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.document_processed = True
                            st.info("Document processed successfully! Summary generated and ready for questions.")
                        else:
                            st.error("Failed to process document.")
                    else:
                        st.error("Failed to extract text from PDF.")

    if not st.session_state.document_processed:
        st.info("Please upload and process a PDF document using the sidebar to get started.")   
    else:
        with st.expander("Paper Summary", expanded=False):
            if st.session_state.paper_summary:
                st.markdown(st.session_state.paper_summary)
            else:
                st.warning("Summary not available.")
        st.subheader("Ask Questions About Your Research Paper")        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"]) 
        if prompt := st.chat_input("Ask a question about the research paper..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})  
            with st.chat_message("user"):
                st.write(prompt)        
            with st.chat_message("assistant"):
                progress_container = st.container()
                execution_trace_container = st.expander("Execution Details", expanded=False)     
                with progress_container:
                    st.info("Initializing AI agents...")
                try:
                    status_placeholder = st.empty()
                    trace_placeholder = execution_trace_container.empty()
                    execution_steps = []
                    def log_step(step):
                        execution_steps.append(step)
                        with trace_placeholder:
                            for i, s in enumerate(execution_steps, 1):
                                st.write(f"{i}. {s}")

                    def status_callback(message):
                        status_placeholder.info(message)
                        log_step(message)                    

                    status_placeholder.info("Initializing AI agents...")
                    log_step("CrewAI agents initialized")
                    crew = create_crew(st.session_state.vectorstore, status_callback)
                    status_placeholder.info("Starting AI analysis...")
                    log_step("CrewAI execution started")
                    status_placeholder.info("Retrieving relevant passages...")
                    retrieved_passages = retrieval_action(prompt, st.session_state.vectorstore)
                    log_step("Retrieved relevant passages from the research paper")
                    status_placeholder.info("Generating comprehensive answer...")

                    inputs = {
                        "user": prompt,
                        "Retriever": retrieved_passages
                    }

                    response = generation_action(inputs)
                    log_step("Generated detailed answer with citations")
                    status_placeholder.info("Analysis complete!")
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.markdown("### AI Analysis Result:")
                    st.write(response)

                    with execution_trace_container:
                        st.markdown("### Execution Summary:")
                        st.info("**Retriever Agent**: Found relevant passages from the research paper")
                        st.info("**Generator Agent**: Created comprehensive answer with citations")
                        st.info(f"**Total Steps**: {len(execution_steps)}")
                        st.markdown("### Detailed Execution Trace:")
                        for i, step in enumerate(execution_steps, 1):
                            st.write(f"{i}. {step}")            

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
     

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        

if __name__ == "__main__":

    main()
