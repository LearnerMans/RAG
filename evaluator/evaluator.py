import os
from typing import List, Dict
from trulens.apps.custom import TruCustomApp
from trulens.core import TruSession
from trulens.core import Feedback
from trulens.providers.openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
from trulens.apps.custom import instrument
from trulens.core import TruSession
from pinecone.grpc import PineconeGRPC as Pinecone



load_dotenv()





class RAG_eval:
    
    def __init__(self,retriever,generator, db_name, reset_database = false):
        self.db_name = db_name
        self.reset_database = reset_database
        self.retriever = retriever # Retriever object
        self.generator = generator  # Generator object
        self.session = self._initiate_session()
        self.index = self._initiate_db()
    
    def _initiate_session( self):
        session = TruSession()
        if(self.reset_database):
            session.reset_database()
        
        return session
    
    def _initiate_db(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(self.db_name)
        return index


    # Instrumented Retrieval methods
    @instrument
    def retrieve(self, query: str) -> List[str]:
        """
        Method to handle document retrieval.
        IMPORTANT: The method name 'retrieve' will be used in selectors
        """
        chunks = self.retriever.get_Chunks(query)
        return chunks
    
    @instrument
    def generate(self, query: str, context: List[str]) -> str:
        """
        Method to handle response generation.
        IMPORTANT: The method name 'generate' will be used in selectors
        """
        formatted_context = "\n".join([str(doc) for doc in context])
        response = self.generator.generate(query, formatted_context)
        return response
    
    @instrument
    def query(self, question: str) -> Dict:
        """
        Main method that orchestrates the RAG pipeline.
        IMPORTANT: Return keys must match selector paths
        """
        context = self.retrieve(question)
        response = self.generate(question, context)
        
        return response





        