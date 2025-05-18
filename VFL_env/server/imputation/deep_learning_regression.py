import flwr as fl
import torch
import functions
from pathlib import Path
import numpy as np
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
    def __init__(self,csv_path, target_feature,final_round, device="cpu",*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = functions.SimpleRegressor(input_dim=8)  # depends on the recived embedding from the nodes (we link the input dimention of each node with a hidden layer of 4 perceptron each.'if there is two participant so we have 8 comming embeddings. ') 
        self.train_target, self.test_target, self.train_indices, self.test_indices = functions.preprocess_server_target(csv_path,target_feature,approche='dl_r',test_size=0.2)

        
        self.device=device
        self.model = self.model.to(self.device)
        self.train_target = self.train_target.to(self.device)
        self.test_target = self.test_target.to(self.device)

        self.test_output = None
        self.test_input = None
        self.final_round = final_round

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

        self.test_input = x.detach()

        # 2. Forward pass
        output = self.model(x)


        # Match only the embeddings related to train_indices
        train_output = output.squeeze()[self.train_indices]
        loss = torch.nn.functional.mse_loss(train_output, self.train_target.squeeze())


        logger.info(f"Server loss: {loss.item()}")
        
        # 3. Backward pass
        loss.backward()

        # 3. Extract gradients per client_id
        sorted_ids = [int(s) for s in sorted_ids]
        sorted_ids = functions.reshape_list_with_none((sorted_ids))
        grads = []
        for i in sorted_ids:
            if i is None:
                grads.append(np.zeros((1,), dtype=np.float32))  # Dummy gradient for missing nodes
            else:
                grads.append(embedding_map[str(i)].grad.clone().numpy())

        self.test_output = output.detach()[self.test_indices].cpu().numpy()

        functions.save_metrics(self.test_output, self.test_target.cpu().numpy(), "../results/dl_regression", server_round)

        if server_round == self.final_round :
            functions.save_model(self.model,model_type='dl_r')

        logger.info(f"Final : Sending gradients: {grads}")

        return ndarrays_to_parameters(grads), {"loss":loss.item()}
    

def start_server():
    strategy = VFLServer(
        csv_path="../target_data/data_r.csv",
        target_feature="respiratory_rate",
        final_round=30,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    
    history = fl.server.start_server(server_address="v_central_server:5000", strategy=strategy, 
        config=fl.server.ServerConfig(num_rounds=30),
        certificates=(
        Path("../certs/ca.pem").read_bytes(),
        Path("../certs/central_server.pem").read_bytes(),
        Path("../certs/central_server.key").read_bytes())
    )
    
    return history 

if __name__ == "__main__" :
    start_server()



