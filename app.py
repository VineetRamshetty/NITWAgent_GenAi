import streamlit as st
from dotenv import load_dotenv
from agent import NITWAgent

load_dotenv()

st.title("NIT Warangal Agent")

agent=NITWAgent()

query=st.text_input("Ask anything about NIT Warangal:", key="query")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        answer = agent.answer(query, k=4)

        st.write(answer)
