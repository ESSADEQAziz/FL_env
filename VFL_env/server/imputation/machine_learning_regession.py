import flwr as fl
import torch
import numpy as np
import functions
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/v_central_server.log"),
    ]
)
logger = logging.getLogger("v_central_server")
logger.info("Strating v_central_server ... ")


class LinearVFLServer(fl.server.strategy.FedAvg):
    def __init__(self, data_path, target_col, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y = functions.preprocess_server_target_ml_r(data_path,target_col)
        self.y = functions.insure_none(self.y)
        self.loss_fn = torch.nn.MSELoss()

        logger.info(f"Initilizing the server with the shape: {self.y.shape}")
        if torch.isnan(self.y).any():
            logger.warning("Warning: NaN values detected in target y, replacing them with 0 to avoid loss and gradients calculus.(consider it as noise)")
        self.y = torch.nan_to_num(self.y, nan=0.0)

    def aggregate_fit(self, server_round, results, failures):
        logger.info(f"Server aggregate_fit at round {server_round}")

        z_map = {}
        for _, res in results:
            node_id = res.metrics["node_id"]
            z_np = parameters_to_ndarrays(res.parameters)[0]
            z_map[node_id] = torch.tensor(z_np, requires_grad=True)
        
        logger.info(f"the received map is : {z_map}")

        # Sort and concatenate partial predictions
        sorted_ids = sorted(z_map.keys())
        z_total = sum(z_map[i] for i in sorted_ids)  # [n_samples, 1]

        logger.info(f"the concatenation of the the partitial predictions z_total is : {z_total}")
        # Compute loss
        loss = self.loss_fn(z_total, self.y) 
        loss.backward()

        functions.save_metrics_ml("../results/ml_regression",server_round,loss.item())

        # Compute and return gradients
        sorted_ids = [int(s) for s in sorted_ids]
        sorted_ids = functions.reshape_list_with_none((sorted_ids)) 
        grads = [z_map[str(i)].grad.clone().numpy() if i is not None else np.zeros((1,), dtype=np.float32) for i in sorted_ids]

        logger.info(f"[Server] Round {server_round} loss: {loss.item():.4f}")
        logger.info(f"the sent gradients are : {grads}")

        return ndarrays_to_parameters(grads), {"loss": loss.item()}

def start_server():
    strategy = LinearVFLServer(
        data_path="./data.csv",
        target_col="anchor_year",
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    fl.server.start_server(server_address="v_central_server:5000", strategy=strategy, 
                           config=fl.server.ServerConfig(num_rounds=20))

if __name__ == "__main__":
    start_server()
