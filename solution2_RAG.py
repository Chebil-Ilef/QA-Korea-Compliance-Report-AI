import logging
import os
import getpass
import subprocess
import pandas as pd

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def configure_logging():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.debug("Logger configured.")


def setup_environment():
    logging.debug("Setting up environment...")
    
    subprocess.call("pip install -r requirements.txt", shell=True)
    subprocess.call("./init.sh", shell=True)
    
    os.environ["LANGSMITH_TRACING"] = "true"
    
    for api_key in ["LANGSMITH_API_KEY", "GROQ_API_KEY"]:
        if not os.environ.get(api_key):
            os.environ[api_key] = getpass.getpass(f"Enter API key for {api_key}: ")
    
    logging.debug("Environment setup complete.")


def load_documents(file_path: str):
    logging.debug(f"Loading documents from {file_path}...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    logging.debug("Documents loaded successfully.")
    return docs


def initialize_vector_store(docs):
    logging.debug("Initializing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = InMemoryVectorStore(embeddings)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)
    logging.debug("Vector store initialized.")
    return vector_store


def retrieve(state: State, vector_store):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State, llm):
    prompt = hub.pull("rlm/rag-prompt")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


def build_graph(vector_store, llm):
    def retrieve_wrapper(state: State):
        return retrieve(state, vector_store)
    
    def generate_wrapper(state: State):
        return generate(state, llm)

    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve_wrapper)
    graph_builder.add_node("generate", generate_wrapper)
    
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")

    return graph_builder.compile()

def process_questions(graph, input_csv, output_file):
    logging.debug("Processing questions...")
    df = pd.read_csv(input_csv, encoding='utf-8', delimiter=';')
    questions = df['question'].tolist()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("question_id;question;answer\n")
    
    for idx, question in enumerate(questions, start=1):
        response = graph.invoke({"question": question})
        with open(output_file, "a") as f:
            f.write(f"{idx};{question};{response['answer']}\n")
    
    logging.debug("Questions processing completed.")


def main():
    configure_logging()
    setup_environment()
    
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    docs = load_documents("./docs/Korea-Follow-Up-Report-2024.pdf")
    vector_store = initialize_vector_store(docs)
    graph = build_graph(vector_store, llm)
    
    process_questions(graph, "docs/korea_aml_questions_all.csv", "./results/solution2_RAG/answers.csv")


if __name__ == "__main__":
    main()
