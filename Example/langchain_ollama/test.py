from constants_local import *

with open(CONFLUENCE_TOKEN_PATH, "r") as ofile:
    CONFLUENCE_TOKEN = ofile.readlines()[0].replace('\n', '')

from customRetriver import CustomRetriver

retriever = CustomRetriver(confluence_domain=CONFLUENCE_DOMAIN,
                            confluence_username='shaswat178@gmail.com',
                            confluence_password=CONFLUENCE_TOKEN
                          )

prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Helpful Answer:"""

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

# Define llm
llm = Ollama(model="gemma2:2b")

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer. Explain in detail.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

print( rag_chain.invoke("what is the use of lightrag?"))