from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def loadDocs():
    textSplitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

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

    db=Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="vectorDB")
    db.persist()

if __name__=="__main__":
    buildDB()
