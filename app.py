import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PAPERS_DIR = "papers"
FAISS_INDEX_PATH = "faiss_index"

PROMPT_TEMPLATE = """
You are a research assistant. Use the context below to answer the question.
Always cite the source document when possible. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:
"""

def load_and_chunk_documents(papers_dir: str):
    loader = DirectoryLoader(papers_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents, split into {len(chunks)} chunks.")
    return chunks

def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}/")
    return vectorstore

def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded.")
    return vectorstore

def build_rag_chain(vectorstore):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0.2
    )
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

def query(chain, question: str):
    result = chain.invoke({"query": question})
    print("\n--- Answer ---")
    print(result["result"])
    print("\n--- Sources ---")
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        print(f"  {source} | Page {page}")
    return result

def main():
    os.makedirs(PAPERS_DIR, exist_ok=True)

    if not os.path.exists(FAISS_INDEX_PATH):
        print("Building FAISS index from papers...")
        chunks = load_and_chunk_documents(PAPERS_DIR)
        if not chunks:
            print(f"No PDFs found in '{PAPERS_DIR}/'. Add research papers and rerun.")
            return
        vectorstore = build_faiss_index(chunks)
    else:
        vectorstore = load_faiss_index()

    chain = build_rag_chain(vectorstore)

    print("\nResearchRAG ready. Type your question (or 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        query(chain, question)

if __name__ == "__main__":
    main()
