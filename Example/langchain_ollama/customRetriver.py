from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from crag.ConfluenceAPIWrapper import ConfluenceAPIWrapper

class CustomRetriver(BaseRetriever, ConfluenceAPIWrapper):
    def _get_relevant_documents(
        self, cql: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        cql = 'text~"*{q}*"'.format(q=cql)
        print( cql )
        confluence = self.load(cql)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        docs = text_splitter.split_documents(confluence)
        vectorstore = InMemoryVectorStore.from_documents(documents=docs, embedding=OllamaEmbeddings(model='gemma2:2b'))
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 6} )
        docs = retriever.invoke(cql)

        print('Releasing vectorstore space - ', vectorstore.delete())
        print( docs )
        return docs