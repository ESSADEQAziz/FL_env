import flwr as fl
import torch
import numpy as np
from pathlib import Path
import json
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

class VFLServer(fl.server.strategy.FedAvg):
    def __init__(self, data_path, target_col, final_round, device="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_y, self.test_y = functions.preprocess_server_target_ml_c(data_path, target_col, test_size=0.2)
        self.device = device
        self.train_y = self.train_y.to(device)
        self.test_y = self.test_y.to(device)
        self.num_classes = self.train_y.shape[1]
        self.final_round = final_round

        logger.info(f"Initilise the server with y_train {self.train_y.shape} y_test {self.test_y.shape}//// {self.train_y} ////{self.test_y}")

        self.model = functions.LinearVFLModel(input_dim=2, output_dim=self.num_classes).to(device) # During training, when you use CrossEntropyLoss, PyTorch automatically applies the softmax inside the loss function.

    def aggregate_fit(self, server_round, results, failures):
        embedding_map = {}
        for _, res in results:
            node_id = res.metrics["node_id"]
            embedding_np = parameters_to_ndarrays(res.parameters)[0]
            logger.warning(f"the received parameters from node {node_id}  /// after : {embedding_np}")
            embedding_map[node_id] = torch.tensor(embedding_np, dtype=torch.float32, requires_grad=True).to(self.device)

        sorted_ids = sorted(embedding_map.keys())
        logger.info(f" (Aggregate fit) round= {server_round} the resulting embedding map is : {embedding_map}")
        embeddings = [embedding_map[cid] for cid in sorted_ids]
        x = torch.cat(embeddings, dim=1)

        train_size = self.train_y.shape[0]
        x_train = x[:train_size]
        x_test = x[train_size:]

        logger.info(f" (round = {server_round}) the x_train {x_train.shape} x_test {x_test.shape} /// {x_train} ////{x_test}")

        output_train = self.model(x_train)
        output_test = self.model(x_test)

        logger.info(f"(round = {server_round}) the output_train {output_train} output_test {output_test}")

        loss_fn = torch.nn.CrossEntropyLoss()
        y_train = torch.argmax(self.train_y, dim=1)
        y_test = torch.argmax(self.test_y, dim=1)

        loss = loss_fn(output_train, y_train)
        acc = (output_test.argmax(dim=1) == y_test).float().mean().item()

        logger.info(f"Server round {server_round} - Loss: {loss.item()} - Accuracy: {acc:.4f}")

        loss.backward()

        if server_round == self.final_round:
            functions.save_model(self.model,model_type='ml_c')

        sorted_ids = [int(s) for s in sorted_ids]
        sorted_ids = functions.reshape_list_with_none(sorted_ids)
        logger.info(f" the sorted_ids are : {sorted_ids}.")
        grads = []
        for i in sorted_ids:
            if i is None:
                grads.append(np.zeros((1,), dtype=np.float32))
            else:
                grads.append(embedding_map[str(i)].grad.clone().cpu().numpy())

        with open("../results/ml_classification/metrics.json", "a") as f:
            json.dump({"round": server_round, "loss": loss.item(), "accuracy": acc}, f)
            f.write("\n")

        return ndarrays_to_parameters(grads), {"loss": loss.item(), "accuracy": acc}


def start_server():
    strategy = VFLServer(
        data_path="../target_data/data_c.csv",
        target_col="marital_status",
        final_round=200,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server(server_address="v_central_server:5000", strategy=strategy, 
        config=fl.server.ServerConfig(num_rounds=200),
        certificates=(
            Path("../certs/ca.pem").read_bytes(),
            Path("../certs/central_server.pem").read_bytes(),
            Path("../certs/central_server.key").read_bytes()))

if __name__ == "__main__":
    start_server()
