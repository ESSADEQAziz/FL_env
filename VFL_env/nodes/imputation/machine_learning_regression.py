import flwr as fl
import functions
import torch
import logging
import os 
import numpy as np


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
    def __init__(self, data_path, feature_names):
        self.x = functions.preprocess_node_data_ml(data_path, feature_names, "regression")  # one or multiple features (categorical features not handled yet.)
        self.w = torch.randn((self.x.shape[1], 1), requires_grad=True)  # one weight per feature
        self.lr = 0.01
        logging.info(f"Initilizing node {NODE_ID} with shape: {self.x.shape}")
        self.x = functions.insure_none(self.x)

    def get_parameters(self, config):
        return [self.w.detach().numpy()]

    def fit(self, parameters, config):
        z = self.x @ self.w  # [n_samples, num_features] x [num_features, 1] = [n_samples, 1]
        logger.info(f"(fit function) node {NODE_ID}, the sent (z=x*w) is : {[z.detach().numpy()]}")
        return [z.detach().numpy()], len(self.x), {"node_id": NODE_ID}


    def evaluate(self, parameters, config):
        dz = parameters[int(NODE_ID)]  # Gradient of loss w.r.t. z
        if np.all(dz == 0) :
            raise ValueError(f"the node {NODE_ID} rceived a none value as gradients.")
        logger.info(f"(evaluation function) node {NODE_ID} the receided parameters are : {parameters}")
        logger.info(f"(evaluation function) node {NODE_ID}, the received gradients are : {dz}")
        
        dz = torch.tensor(dz, dtype=torch.float32)
        
        z = self.x @ self.w
        z.backward(dz)
        
        logger.info(f"(evaluation function) the last weights for node {NODE_ID} are : {self.w.shape} {self.w}")

        with torch.no_grad():
            self.w -= self.lr * self.w.grad
        self.w.grad.zero_()

        logger.info(f"(evaluation function) the updated weights for node {NODE_ID} are : {self.w.shape} {self.w}")

        return 0.0, len(self.x), {}

if __name__ == "__main__":

    features = ["anchor_age","valuenum"]
    data_path = "../data/data.csv"

    fl.client.start_numpy_client(server_address="v_central_server:5000", client=VFLClient(data_path,features))



