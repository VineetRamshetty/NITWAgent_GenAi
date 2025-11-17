from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

load_dotenv()

template="""
You are NITW-InfoGPT. Use ONLY the provided context from the NIT Warangal dataset.
If the answer is not found in the context, reply:
"I don't have this information in the NITW dataset."

Context: {context}

Question: {question}

Answer:
"""

class NITWAgent:
    def __init__(self, model_repo="meta-llama/Llama-3.1-8B-Instruct"):
        self.embeddings=HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2", 
            model_kwargs={"device": "cpu"}
        )
        
        self.db=Chroma(persist_directory="vectorDB", embedding_function=self.embeddings)

        llm=HuggingFaceEndpoint(
            repo_id=model_repo,
            task="conversational",
            temperature=0.2,
            max_new_tokens=256
        )

        self.llm=ChatHuggingFace(llm=llm)

        self.prompt_template=PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

    def answer(self, question, k=4):
        docs=self.db.similarity_search(question, k=k)
        if not docs:
            return "No relevant information found in the NITW vectorDB."

        context="\n\n".join(doc.page_content for doc in docs)

        final_prompt = self.prompt_template.format(context=context, question=question)

        response=self.llm.invoke(final_prompt)

        return response.content
