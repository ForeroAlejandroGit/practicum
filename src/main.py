import streamlit as st

st.title("Super simple title")

st.header("This is a header")

st.subheader("This is a subheader")

st.write("This is a text")

st.markdown("This is a _markdown_")

code_example = """
def hello_world():
    print("Hello World")
"""

st.code(code_example, language="python")

st.divider()