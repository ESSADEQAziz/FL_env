import logging
import numpy as np
import flwr as fl
import functions
from flwr.server.strategy import FedAvg

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
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.param_log = []  # [(round, parameters)]
        
    def aggregate_fit(self, server_round, results, failures):
        """Custom aggregation of (a, b) parameters."""

        if failures:
            logger.warning(f"Failures: {len(failures)} clients failed to send updates")

        # Initialize lists
        a_values = []
        b_values = []
        total_samples = 0

        for client, fit_res in results:
            parameters = fit_res.parameters  # Extract parameters (a, b)
            num_samples = fit_res.num_examples  # Extract number of training samples

            # Convert to NumPy arrays
            params = fl.common.parameters_to_ndarrays(parameters)

            # Ensure parameters are lists or 1D arrays
            if isinstance(params, list) and len(params) == 2:
                a, b = params
            elif isinstance(params, np.ndarray) and params.size == 2:
                a, b = params.tolist()  # Convert single array to list
            else:
                logger.error(f"Unexpected parameter format: {params}")
                continue

            logger.info(f"the received parameters : a = {a} and b = {b}")

            a_values.append((a, num_samples))
            b_values.append((b, num_samples))
            total_samples += num_samples
        

        # Compute weighted averages
        if total_samples == 0:
            logger.error("No valid data received from clients.")
            return None, {}
        
        # Compute Weighted Averages (Each node's a (or b) is multiplied by the number of samples (w) it trained on)
        avg_a = sum(a * w for a, w in a_values) / total_samples
        avg_b = sum(b * w for b, w in b_values) / total_samples

        # Convert back to Flower parameters format
        aggregated_parameters = fl.common.ndarrays_to_parameters([np.array(avg_a), np.array(avg_b)])

        logger.info(f"Round {server_round}: Aggregated a={avg_a}, b={avg_b}")

        self.param_log.append((server_round, [avg_a, avg_b]))

       # The returned results after the evaluations within the distributed server test data
        return aggregated_parameters, {"aggregated_MSE":None}
    


    
    def evaluate(self, server_round, parameters):
       """Send the round number and aggregated loss to nodes."""
       # The returned results after the evaluations within the centralised server test data
       return None, {"aggregated_MSE":None}
    
def fit_config(server_round):
    return {
        "server_round": server_round
    }

def start_server():
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=fit_config)
    
    history = fl.server.start_server(server_address="central_server:5000", strategy=strategy, config=fl.server.ServerConfig(num_rounds=5))
    return history ,strategy.param_log
    
if __name__ == "__main__":
    history,strategy_param = start_server() 
    history.losses_distributed
    functions.evaluate_ml_values(history,strategy_param,logger)
