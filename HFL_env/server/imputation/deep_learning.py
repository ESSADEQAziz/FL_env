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

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = []
        self.total_rounds=5

    def aggregate_fit(self, server_round,results,failures):
        for client, fit_res in results:
            parameters = fit_res.parameters  
            num_samples = fit_res.num_examples  

        logger.info(f"(aggregate_fit) the received parameters are : {parameters.__sizeof__} with num_exemples {num_samples} ")

        # Call aggregate_fit from base class
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if functions.save_dl_model(aggregated_parameters,server_round,5,f"../results/dl_results/agg_model/model_round{server_round}.pt"):
            logger.info("The model saved successfully.")
                
        return aggregated_parameters, metrics



def start_server():
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    
    history = fl.server.start_server(server_address="central_server:5000", strategy=strategy, 
        config=fl.server.ServerConfig(num_rounds=5),#if you change the num_round, change it also within the save_dl_model() function
        certificates=(
        Path("../certs/ca.pem").read_bytes(),
        Path("../certs/central_server.pem").read_bytes(),
        Path("../certs/central_server.key").read_bytes())
    )
    return history 
    
if __name__ == "__main__":
    functions.evaluate_dl_values(start_server()) 

