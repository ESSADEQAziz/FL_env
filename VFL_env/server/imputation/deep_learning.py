import flwr as fl
import torch
import functions
import logging
from flwr.common import parameters_to_ndarrays
from flwr.common import ndarrays_to_parameters

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

class VFLServer(fl.server.strategy.FedAvg):
    def __init__(self,csv_path, target_feature, device="cpu",*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = functions.SimpleRegressor(input_dim=8)  # depends on the recived embedding from the nodes (we link the input dimention of each node with a hidden layer of 4 perceptron each.'if there is two participant so we have 8 comming embeddings. ') 
        self.target = functions.preprocess_server_num_target(csv_path,target_feature)

        self.device=device
        self.model = self.model.to(self.device)
        self.target = self.target.to(self.device)

    def aggregate_fit(self, server_round, results, failures):

        logger.info(f"Server aggregate_fit at round {server_round}")

        # 1. Create a map from client_id -> embedding tensor
        embedding_map = {}
        for _, res in results:
            # Extract the node/client ID
            node_id = (res.metrics["node_id"])

            # Convert Parameters object (bytes) back to NumPy arrays
            embedding_np = parameters_to_ndarrays(res.parameters)[0]  # Only one tensor sent per client

            # Convert to torch.Tensor, enable gradient, and move to device
            emb = torch.tensor(embedding_np, dtype=torch.float32, requires_grad=True).to(self.device)

            logger.info(f"the getting parameters from node {node_id} are  (size : {emb.shape}) : {emb}")
            # Add to map
            embedding_map[node_id] = emb

        # 2. Concatenate embeddings in client_id order (sorted for reproducibility)
        sorted_ids = sorted(embedding_map.keys())
        embeddings = [embedding_map[cid] for cid in sorted_ids]

        logger.warning(f"the sorted embeddings are (size {len(embeddings)}): {embeddings}")

            # 2. Concatenate embeddings
        x = torch.cat(embeddings, dim=1)  # Shape: [batch_size, total_embedding_dim]

        logger.warning(f"the x shape is {x.shape}")
 

        # 2. Forward pass
        output = self.model(x)
        loss = torch.nn.functional.mse_loss(output.squeeze(), self.target)
        logger.info(f"Server loss: {loss.item()}")
        
        # 3. Backward pass
        loss.backward()

        # 3. Extract gradients per client_id
        grad_map = {
            cid: emb.grad.clone().detach().cpu().numpy()
            for cid, emb in zip(sorted_ids, embeddings)
        }

        logger.info(f"Final : Sending gradients: {list(grad_map.items())}")
        final_grads=ndarrays_to_parameters(grad_map.values())
        return final_grads, {"loss":loss.item()}
    

def start_server():
    strategy = VFLServer(
        csv_path="./data.csv",
        target_feature="anchor_year",
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    
    history = fl.server.start_server(server_address="v_central_server:5000", strategy=strategy, 
        config=fl.server.ServerConfig(num_rounds=100))
    
    return history 

if __name__ == "__main__" :
    start_server()



