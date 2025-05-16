import torch
import os
import numpy as np
import json 
from pathlib import Path
import flwr as fl
import functions
import logging
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

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
    def __init__(self, csv_path, target_feature,final_round, device="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_y, self.test_y = functions.preprocess_server_target(csv_path, target_feature,approche='dl_c',test_size=0.2)
        
        self.device = device
        self.train_y = self.train_y.to(device)
        self.test_y = self.test_y.to(device)
        self.num_classes = self.train_y.shape[1]
        self.final_round = final_round
        self.model = functions.SimpleClassifier(input_dim=20, num_classes=self.num_classes).to(device)# depends on the recived embedding from the nodes (we link the input dimention of each node with a hidden layer of 4 perceptron each.'if there is two participant so we have 8 comming embeddings. ')
        
      
    def aggregate_fit(self, server_round, results, failures):
        embedding_map = {}
        for _, res in results:
            node_id = res.metrics["node_id"]
            embedding_np = parameters_to_ndarrays(res.parameters)[0]
            embedding_map[node_id] = torch.tensor(embedding_np, dtype=torch.float32, requires_grad=True).to(self.device)

        sorted_ids = sorted(embedding_map.keys())
        embeddings = [embedding_map[cid] for cid in sorted_ids]
        x = torch.cat(embeddings, dim=1)

        # Split embeddings the same way as y
        train_size = self.train_y.shape[0]
        x_train = x[:train_size]
        x_test = x[train_size:]

        output_train = self.model(x_train)
        output_test = self.model(x_test)

        loss_fn = torch.nn.CrossEntropyLoss()
        y_train = torch.argmax(self.train_y, dim=1)
        y_test = torch.argmax(self.test_y, dim=1)

        loss = loss_fn(output_train, y_train)
        acc = (output_test.argmax(dim=1) == y_test).float().mean().item()

        logger.info(f"Server round {server_round} - Loss: {loss.item()} - Accuracy: {acc:.4f}")

        loss.backward()

        if server_round == self.final_round:
            os.makedirs("../results/dl_classification/", exist_ok=True)
            torch.save(self.model.state_dict(), "../results/dl_classification/final_vfl_model.pth")

        # 3. Extract gradients per client_id
        sorted_ids = [int(s) for s in sorted_ids]
        sorted_ids = functions.reshape_list_with_none((sorted_ids))
        grads = []
        for i in sorted_ids:
            if i is None:
                grads.append(np.zeros((1,), dtype=np.float32))  # Dummy gradient for missing nodes
            else:
                grads.append(embedding_map[str(i)].grad.clone().numpy())

        # Save results per round
        with open("../results/dl_classification/metrics.json", "a") as f:
            json.dump({"round": server_round, "loss": loss.item(), "accuracy": acc}, f)
            f.write("\n")

        return ndarrays_to_parameters(grads), {"loss": loss.item(), "accuracy": acc}
    
      
def start_server():
    strategy = VFLServer(
        csv_path="../target_data/data_c.csv",
        target_feature="insurance",
        final_round=30,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5
    )

    fl.server.start_server(
        server_address="v_central_server:5000",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=30), # change also the final_round attribute within the strategy.
        certificates=(
            Path("../certs/ca.pem").read_bytes(),
            Path("../certs/central_server.pem").read_bytes(),
            Path("../certs/central_server.key").read_bytes())
    )

if __name__ == "__main__":
    start_server()
