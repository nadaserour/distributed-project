class Scheduler:
    def __init__(self, load_balancer):
        self.lb = load_balancer

    def handle_request(self, request):
        print(f"[Scheduler] Dispatching request {request.id}")
        response = self.lb.dispatch(request)
        return response 