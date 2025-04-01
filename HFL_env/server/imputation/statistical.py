import logging
import flwr as fl
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/central_server.log"),
    ]
)
logger = logging.getLogger("central_server")


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, aggregate_fn, **kwargs):
        super().__init__(**kwargs)
        self.aggregate_fn = aggregate_fn

    def aggregate(self, results: Dict):
        # Custom aggregation function
        return self.aggregate_fn(results)
    

def start_server():
    def aggregate(results: Dict):
        logger.info(f'the result comming from the nodes are : {results}')
        # Aggregate the mean and median values
        means = [r[1][0] for r in results]
        medians = [r[1][1] for r in results]
        aggregated_mean = sum(means) / len(means)
        aggregated_median = sum(medians) / len(medians)
        return [aggregated_mean, aggregated_median]

    # Create the custom FedAvg strategy
    strategy = CustomFedAvg(
        aggregate_fn=aggregate,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )

    """
    the flower server get the parameters from a random node and sent them to all nodes as the defaults
    parameters, then the real process of aggragation is applyed after getting all parameters 
    from all nodes and send the aggregated parameters back to all nodes for evaluation. 
    """

    # Start the federated learning server 
    fl.server.start_server(
        server_address="central_server:5000",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )

if __name__ == "__main__":
    # Start Flower client
    start_server()
    