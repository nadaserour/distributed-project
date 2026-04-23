from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test

def main():
    # Create GPU workers
    workers = [GPUWorker(i) for i in range(4)] # simulate 4 GPUs
    
    # Load Balancer
    lb = LoadBalancer(workers)
    
    # Scheduler
    scheduler = Scheduler(lb)
    
    # Run simulation
    run_load_test(scheduler, num_users=1000)

if __name__ == "__main__":
    main() 