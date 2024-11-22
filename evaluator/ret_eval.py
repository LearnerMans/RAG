import pandas as pd
from trulens.core import TruSession
from trulens.core import Feedback
from trulens.core.schema.select import Select
from trulens.feedback import GroundTruthAgreement
from trulens.providers.openai import OpenAI as fOpenAI
from trulens.apps.custom import TruCustomApp
from trulens.apps.custom import instrument
from utils.chunk_scorer import score_retrieved_chunks



class retriever_evaluator:
    """
    
    """

    def __init__(self,name, ground_truth, rag_app , reset_db = False):
        self.ground_truth = self._init_ground_truth(ground_truth) 
        self.session = self._init_db(reset_db)
        self.feedback = self._feedback_init()
        self.name = name
        self.rag_app = rag_app
        self.tru_app = self._init_app()

### Move the addition of the scores  to prepare ground truth 
    def _init_ground_truth(self,ground_truth):
        queries =  ground_truth["query"]
        expected_responses =  ground_truth["expected_response"]
        expected_chunks = ground_truth["expected_chunks"]
        
        for i in range(len(queries)):
            expected_score = score_retrieved_chunks(queries[i],expected_chunks[i],expected_responses[i])
            expected_chunks[i] = {"text":expected_chunks[i], "title":expected_chunks[i], "score":expected_score}
        return pd.DataFrame(ground_truth)

    def _init_db(self, reset_db):
        session = TruSession()
        session.reset_database() if reset_db else None
        return session
    
    def _feedback_init(self):
        arg_query_selector = (
            Select.RecordCalls.retrieve_and_generate.args.query
        )  # 1st argument of retrieve_and_generate function
        arg_retrieval_k_selector = (
            Select.RecordCalls.retrieve_and_generate.args.k
        )  # 2nd argument of retrieve_and_generate function

        arg_completion_str_selector = Select.RecordCalls.retrieve_and_generate.rets[
            0
        ]  # 1st returned value from retrieve_and_generate function
        arg_retrieved_context_selector = Select.RecordCalls.retrieve_and_generate.rets[
            1
        ]  # 2nd returned value from retrieve_and_generate function
        arg_relevance_scores_selector = Select.RecordCalls.retrieve_and_generate.rets[
            2
        ]  # last returned value from retrieve_and_generate function

        f_ir_hit_rate = (
            Feedback(
                GroundTruthAgreement(self.data, provider=fOpenAI()).ir_hit_rate,
                name="IR hit rate",
            )
            .on(arg_query_selector)
            .on(arg_retrieved_context_selector)
            .on(arg_retrieval_k_selector)
        )

        f_ndcg_at_k = (
            Feedback(
                GroundTruthAgreement(self.data, provider=fOpenAI()).ndcg_at_k,
                name="NDCG@k",
            )
            .on(arg_query_selector)
            .on(arg_retrieved_context_selector)
            .on(arg_relevance_scores_selector)
            .on(arg_retrieval_k_selector)
        )


        f_recall_at_k = (
                Feedback(
                GroundTruthAgreement(self.data, provider=fOpenAI()).recall_at_k,
                name="Recall@k",
            )
            .on(arg_query_selector)
            .on(arg_retrieved_context_selector)
            .on(arg_relevance_scores_selector)
            .on(arg_retrieval_k_selector)
        )
        f_groundtruth_answer = (
            Feedback(
            GroundTruthAgreement(self.data).agreement_measure,
            name="Ground Truth answer (semantic similarity)",
            )
            .on(arg_query_selector)
            .on(arg_completion_str_selector))
        return [f_ir_hit_rate, f_ndcg_at_k, f_recall_at_k, f_groundtruth_answer]

    def _init_app(self):
        tru_app = TruCustomApp(
            self.rag_app,
            app_name=self.name,
            feedbacks=self.feedback,
            )
            return tru_app
    def run(self ):
        queries = self.ground_truth["query"]
        for query in queries:
            with tru_app as recording:
                resp = self.rag_app.retrieve_and_generate(query)



class rag_app:
    def __init__(self, retriever, generator, expected_responses):
        self.retriever = retriever
        self.generator = generator
        self.expected_responses = expected_responses






    @instrument
    def retrieve_and_generate(self, query, k=10):
        chunks = self.retrieve.get_Chunks(query)
        chunks = [chunk["metadata"]["text"] for chunk in chunks]
        response = self.generator.generate(query, chunks)
        scores = [ score_retrieved_chunks(query, chunks , self.expected_responses[i])  for i,chunk in enumerate(chunks)]
        return response, chunks, scores


    

