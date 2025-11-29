from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def loadDocs():
    textSplitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    docs=[]
    for p in sorted(Path("data").glob("*.txt")):
        text=p.read_text(encoding="utf-8")
        chunks=textSplitter.create_documents([text])
        docs.extend(chunks)
    return docs

def buildDB():
    docs=loadDocs()

    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2", 
        model_kwargs={"device": "cpu"}
    )

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")


if __name__=="__main__":
    buildDB()
