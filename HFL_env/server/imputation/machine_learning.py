import logging
import flwr as fl
import functions
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
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def aggregate_fit(self, server_round, results, failures):
        """Custom aggregation of (a, b) parameters."""

        if failures:
            logger.warning(f"Failures: {len(failures)} clients failed to send updates")
            raise ValueError(f"Failure : {failures}")

        # Initialize lists
        weights_list = []
        biases_list = []
        total_samples = 0

        for client, fit_res in results:
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            num_samples = fit_res.num_examples
            input_dim = fit_res.metrics.get("input_dim")

            logger.info(f"the number of sample are : {num_samples} , the input_dim {input_dim} and the comming parameters are :{params}")
            if len(params) != 2:
                logger.error(f"Unexpected parameter format: {params}")
                continue

            weight, bias = params
            weights_list.append((weight, num_samples))
            biases_list.append((bias, num_samples))
            total_samples += num_samples
        logger.info(f"the length of the weights list is : {len(weights_list)} / biases length {len(biases_list)} / num_samples {total_samples}")
        # Compute weighted averages
        if total_samples == 0:
            logger.error("No valid data received from nodes.")
            return None, {}
        
        aggregated_parameters,metrics = super().aggregate_fit(server_round,results,failures)
        logger.info(f"the aggregated parameters are {aggregated_parameters} /// metrics {metrics}")

        if server_round == 20 : 
            functions.save_model(aggregated_parameters,server_round,input_dim,"ml_r")
            logger.info("The model saved successfully.")
        
        return aggregated_parameters, metrics
    
    
    def evaluate(self, server_round, parameters):
       """Send the round number and aggregated loss to nodes."""
       # The returned results after the evaluations within the centralised server test data
       return 0.0, {}
    
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
    
    history = fl.server.start_server(server_address="central_server:5000", strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=20),#if you change the num_round, change it also within the save_dl_model() function
        certificates=(
        Path("../certs/ca.pem").read_bytes(),
        Path("../certs/central_server.pem").read_bytes(),
        Path("../certs/central_server.key").read_bytes()
    ))
    return history 
    
if __name__ == "__main__":
    functions.save_metrics(start_server(),"ml_r") 
