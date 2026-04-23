import time
from llm.inference import run_llm
from rag.retriever import retrieve_context

class GPUWorker:
    def __init__(self, id):
        self.id = id

    def process(self, request):
        start = time.time()
        print(f"[Worker {self.id}] Processing request {request.id}")
        
        # RAG Step
        context = retrieve_context(request.query)
        
        # LLM Step
        result = run_llm(request.query, context)
        
        latency = time.time() - start
        return {
            "id": request.id,
            "result": result,
            "latency": latency
        } 