"""
    Utility to call Confluence search and get_page API
"""
import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional
import traceback

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSectionSplitter

class ConfluenceAPIWrapper(BaseModel):
    """
        Wrapper around Confluence API

        To use, you should have the ``atlassian-python-api`` python package installed.
        https://atlassian-python-api.readthedocs.io/
        This wrapper will use the Confluence API to conduct searches and
        fetch pages. By default, it will return the pages of the top-k results.
        If the query is in the form of confluence query langauge
        (see https://developer.atlassian.com/server/confluence/advanced-searching-using-cql/), 
        it will return the pages corresponding to the page id.
        It splits the pages(int HTML format) into Document using langchain's HTMLSectionSplitter.
        Set headers_to_split_on for more control over splitting of text. It also limits the Document content 
        by doc_content_chars_max. Set doc_content_chars_max=None if you don't want to limit the content size.

        Attributes:
            top_k_results: number of the top-scored pages used for the cql.
            continue_on_failure (bool): If True, continue loading other URLs on failure.
            load_max_pages: a limit to the number of loaded pages
            page_content_chars_max: an optional cut limit for the length of a page's content post html split on headers.

        Example:
            .. code-block:: python

                from crag.ConfluenceAPIWrapper import ConfluenceAPIWrapper
                confluence = ConfluenceAPIWrapper(
                    confluence_domain = 'https://www.yourdomain.com',
                    confluence_username = 'your_username',
                    confluence_password = 'your_password_or_token',
                    top_k_results = 3,
                    load_max_pages = 3,
                    doc_content_chars_max
                )
                confluence.run("light rag")
    """
    confluence_domain: str
    confluence_username:str
    confluence_password:str
    top_k_results: int = 3
    continue_on_failure: bool = False
    load_max_pages: int = 100
    page_content_chars_max: Optional[int] = 4000
    html_splits_on_headers: Optional[List[tuple]] = [
        ('h1', 'Header 1'),
        ('h2', 'Header 2'),
        ('h3', 'Header 3'),
        ('h4', 'Header 4'),
        ('h5', 'Header 5'),
        ('h6', 'Header 6'),
        ('p', 'Paragraph'),
        ('table', 'Table')
    ]

    confluence: Optional[Any] = None

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            from atlassian import Confluence
            values['confluence'] = Confluence(values['confluence_domain'],
                                               username=values['confluence_username'],
                                               password=values['confluence_password']
                                            )
            values["arxiv_search"] = values['confluence'].cql
        except ImportError:
            raise ImportError(
                "Could not import atlassian-python-api python package. "
                "Please install it with `pip install atlassian-python-api`."
            )
        return values
    def load( self, cql:str ) -> Iterator[Document]:
        return self.lazy_load( cql )
    def lazy_load(self, cql: str, html_splits_on_headers: Optional[List[tuple]]=None) -> Iterator[Document]:
        """
            Run Confluence search and get the pages texts.
            See https://atlassian-python-api.readthedocs.io/confluence.html#cql

            Returns: Confluence pages with the document.page_content in text format

            Performs an confluence search, gets the HTML formatted pages, split's the pages on HTML headers 
            and returns it as Document

            Args:
                cql: a cql formatted search query
        """
        try:
            from atlassian import Confluence
        except ImportError:
            raise ImportError(
                "Could not import atlassian-python-api python package. "
                "Please install it with `pip install atlassian-python-api`."
            )
        self.confluence = Confluence(self.confluence_domain, username=self.confluence_username, password=self.confluence_password)
        html_splits_on_headers = self.html_splits_on_headers if html_splits_on_headers == None else html_splits_on_headers
        try:
            results = self.confluence.cql(cql)
            # Add capabilities to select the type of content
            results = [ res for res in results['results'] if 'content' in res and res['content']['type'] == 'page' ][ : self.load_max_pages ]
            
            # Make sure ['content']['title'] exists in all the content types
            meta_data = [ {'Title': res['content']['title'], 'source': res['content']['id']} for res in results ]

            results = [ self.confluence.get_page_by_id( res['content']['id'], expand='body.storage' ) for res in results ]
            
            results = [ page['body']['storage']['value'] for page in results if page ]
            
            results = [
                Document(
                    page_content=page,
                    metadata=m_data
                )
                for page, m_data in zip( results, meta_data )
                
            ]

            results = HTMLSectionSplitter(html_splits_on_headers).split_documents( results )
            return results
        except Exception as e:
            print('[ConfluenceAPIWrapper][lazy_load] ', e)
            print(traceback.format_exc())
            return

