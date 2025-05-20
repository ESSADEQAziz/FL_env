# All paths based on the launch.sh file location
import logging
import functions
import flwr as fl
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/central_server.log"),
    ]
)
logger = logging.getLogger("central_server")
logger.info("Starting central server ...")

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self,*args, **kwargs):
        super().__init__(*args,**kwargs)

    
    # we can delete this function and flwr will do aggregation(Avg) automaticlly
    def aggregate_fit(self, server_round,results,failures):
        logger.info(f"(aggregate_fit) The result coming from the nodes are : {results}")

        for client, fit_res in results:
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            
            num_samples = fit_res.num_examples
            logger.warning(f"params are : {params} / and num_samples are : {num_samples}")
        

        # Call aggregate_fit from base class
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        metrics["aggregated_mean_med"] = fl.common.parameters_to_ndarrays(aggregated_parameters)

        # We can send nothing what ever because here is just one round 
        return aggregated_parameters, metrics
    

def start_server():

    # Create the custom FedAvg strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    """
    the flower server get the parameters from a random node and sent them to all nodes as the defaults
    parameters, then the real process of aggragation is applyed after getting all parameters 
    from all nodes and send the aggregated parameters back to all nodes for evaluation. 
    """

    # Start the federated learning server 
    history = fl.server.start_server(
        server_address="central_server:5000",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
         certificates=(
        Path("../certs/ca.pem").read_bytes(),
        Path("../certs/central_server.pem").read_bytes(),
        Path("../certs/central_server.key").read_bytes()
    ))
    return history

if __name__ == "__main__":
    # Start Flower client
    functions.save_history_metrics(start_server(),indx="stat")