import logging
import flwr as fl
from pathlib import Path
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate(self, results: Dict):
        logger.info(f'The result coming from the nodes: {results}')
        means = [r[1][0] for r in results]
        medians = [r[1][1] for r in results]
        aggregated_mean = sum(means) / len(means)
        aggregated_median = sum(medians) / len(medians)
        return [aggregated_mean, aggregated_median]
    

def start_server():

    # Create the custom FedAvg strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
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
         certificates=(
        Path("../certs/ca.pem").read_bytes(),
        Path("../certs/central_server.pem").read_bytes(),
        Path("../certs/central_server.key").read_bytes()
    )
    )

if __name__ == "__main__":
    # Start Flower client
    start_server()