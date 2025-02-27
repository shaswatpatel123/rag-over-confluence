from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .ConfluenceAPIWrapper import ConfluenceAPIWrapper

class ConfluenceRetriever(BaseRetriever, ConfluenceAPIWrapper):
    """`Confluence` retriever.

    Setup:
        Install ``atlassian-python-api``:

        .. code-block:: bash

            pip install -U atlassian-python-api

    Key init args:
        confluence_domain: str
            domain you are using to host confluence
        confluence_username: str
            your username or username used to generate the API token
        confluence_password: str
            token key
        load_max_pages: int
            maximum number of pages to load based on cql

    Instantiate:
        .. code-block:: python

            from crag.ConfluenceRetriever import ConfluenceRetriever

            retriever = ConfluenceRetriever(
                confluence_domain = 'https://www.yourdomain.com',
                confluence_username = 'your_username',
                confluence_password = 'your_password_or_token',
                top_k_results = 3,
                load_max_pages = 3
            )

    Usage:
        .. code-block:: python

            docs = retriever.invoke("What is lightrag?")
            docs[0].metadata

        .. code-block:: none

            [Document(id='bbfccb85-2811-4930-ac72-f9f0662051b5', metadata={'Title': 'LightRag Home', 'source': '2719967', 'Paragraph': ...), ...]
            "
    """  # noqa: E501

    def _get_relevant_documents(
        self, cql: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        cql = 'text~"*{q}*"'.format(q=cql)
        return self.load(cql)