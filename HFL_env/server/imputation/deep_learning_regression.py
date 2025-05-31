import logging
import flwr as fl
from pathlib import Path
import functions

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
    def __init__(self,final_round, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_round=final_round

    def aggregate_fit(self, server_round,results,failures):
        for client, fit_res in results:
            parameters = fit_res.parameters  
            num_samples = fit_res.num_examples  
            input_dim = fit_res.metrics.get("input_dim")

        logger.info(f"(aggregate_fit) the received parameters are : {parameters.__sizeof__} //// {input_dim}  /// with num_exemples {num_samples} ")

        # Call aggregate_fit from base class
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        logger.warning(f"the aggregated parameters are : {aggregated_parameters.__sizeof__} /// {aggregated_parameters}")

        if server_round == self.final_round :
            functions.save_model(aggregated_parameters,server_round,input_dim,"dl_r")
            logger.info("The model saved successfully.")
                
        return aggregated_parameters, metrics
    
    def evaluate(self, server_round, parameters):
        logger.info(f"(evaluation function)the received parameters are : {parameters}")
        return 0.0,{'mse' : 0.0}


def start_server():
    strategy = CustomFedAvg(
        final_round=10,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )
    
    history = fl.server.start_server(server_address="central_server:5000", strategy=strategy, 
        config=fl.server.ServerConfig(num_rounds=10),#if you change the num_round, change it also within the final_round param inside the strategy
        certificates=(
        Path("../certs/ca.pem").read_bytes(),
        Path("../certs/central_server.pem").read_bytes(),
        Path("../certs/central_server.key").read_bytes())
    )
    return history
    
if __name__ == "__main__":
    functions.save_history_metrics(start_server(),"dl_r") 

