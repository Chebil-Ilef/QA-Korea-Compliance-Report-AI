import logging
import os
import getpass
import subprocess
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
import pandas as pd

from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


logging.basicConfig(level=logging.DEBUG)
logging.debug("Starting script execution")

def setup_environment():
    logging.debug("Setting up environment...")
    os.environ["LANGSMITH_TRACING"] = "true"
    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for langsmith: ")

    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for groq: ")


    logging.debug("Environment setup complete")



def main():
    logging.debug("Main function started")

    setup_environment()

    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")    
    vector_store = InMemoryVectorStore(embeddings)

    loader = PyPDFLoader("./docs/Korea-Follow-Up-Report-2024.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)


    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])

        return {"context": retrieved_docs}


    def generate(state: State):
        prompt = hub.pull("rlm/rag-prompt")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()


    # csv for results
    output_file = "./results/solution2_RAG/answers.csv"
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("question_id;question;answer\n")

    # pass all questions to the graph
    df = pd.read_csv("docs/korea_aml_questions_all.csv", encoding='utf-8', delimiter=';')
    questions = df['question'].tolist() 

    for idx, question in enumerate(questions, start=1):
        response = graph.invoke({"question": question})
        # save the answer to a file
        with open(output_file, "a") as f:
            f.write(f"{idx};{question};{response['answer']}\n")




if __name__ == "__main__":
    main()
