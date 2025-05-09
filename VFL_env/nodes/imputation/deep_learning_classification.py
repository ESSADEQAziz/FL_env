import os
import numpy as np
import torch
import flwr as fl
import functions
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/nodes.log"),
    ]
)
logger = logging.getLogger("node")
NODE_ID = os.environ.get("NODE_ID", "0")
logger.info(f"Starting node {NODE_ID} ... ")

class VFLClient(fl.client.NumPyClient):
    def __init__(self, csv_path, features, device="cpu"):
        self.data = functions.preprocess_features(csv_path, features,"dl_c").to(device)
        self.data = functions.insure_none(self.data)
        self.embedding_size = self.data.shape[1]
        self.encoder = functions.ClientEncoder(input_dim=self.embedding_size).to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.01)
        logger.info(f"Node {NODE_ID} initialized with input size : {self.embedding_size}")

    def get_parameters(self, config):
        logger.info(f"(get function) node {NODE_ID}.")
        return []

    def fit(self, parameters, config):
        embeddings = self.encoder(self.data)
        logger.info(f" (fit function) Node {NODE_ID} sent embeddings with shape {embeddings.shape}.embeddings : {[embeddings.detach().numpy()]}")
        return [embeddings.detach().numpy()], len(self.data), {"node_id": NODE_ID}

    def evaluate(self, parameters, config):
        logger.info(f"(evaluation function) node {NODE_ID}, the received parameters are : {parameters}")
        grad_np = parameters[int(NODE_ID)] 

        if np.all(grad_np == 0) :
            raise ValueError(f"the node {NODE_ID} rceived a none value as gradients.")
        
        grad = torch.tensor(grad_np, dtype=torch.float32).to(self.device)

        embeddings = self.encoder(self.data)
        embeddings.backward(grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return 0.0, len(self.data), {}

if __name__ == "__main__":

    # before testing , make sure that the numbers of sum(embeddings) from nodes == the input dimention of the global model within the server (because it may be some categorical features can scale dimention due to one hot encoder)
    target_features = ["gender","marital_status","valuenum","los"]
    target_table = "../data/data.csv"

    private_key = Path(f"../auth_keys/node{NODE_ID}_key")
    public_key = Path(f"../auth_keys/node{NODE_ID}_key.pub")
    ca_cert = Path(f"../certs/ca.pem").read_bytes()

    fl.client.start_numpy_client(
        server_address="v_central_server:5000",
        client=VFLClient(csv_path=target_table, features=target_features),
        root_certificates=ca_cert,
        insecure=False,)
      # authentication_keys=(private_key, public_key),) authentication_keys are not supported in the default gRPC+TLS transport, This feature (authentication_keys) only works with the experimental HTTP/2-based transport layer
    
    
