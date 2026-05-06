import streamlit as st
import requests

st.set_page_config(
    page_title="Ask My Resume",
    page_icon="📄"
)

st.title("📄 Resume Bot")

st.write("Ask questions from the resume")

query = st.text_input("Enter your question")

if st.button("Get"):

    if query.strip() == "":
        st.warning("Please enter a question")

    else:

        with st.spinner("Generating answer..."):

            response = requests.post(
                "http://127.0.0.1:8000/ask",
                params={"query": query}
            )

            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Citations")

            for citation in data["citations"]:
                st.write(
                    f"📄 {citation['source']} | Page {citation['page']}"
                )

            st.subheader("Retrieved Context")

            for i, context in enumerate(data["contexts"]):
                with st.expander(f"Chunk {i+1}"):
                    st.write(context)