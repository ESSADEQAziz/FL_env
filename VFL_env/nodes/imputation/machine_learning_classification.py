import flwr as fl
import functions
import torch
import logging
import os 
from pathlib import Path
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
    def __init__(self, data_path, features_names):
        self.x,self.used_features = functions.preprocess_features(data_path, features_names, NODE_ID,"ml_c")  # one or multiple features 
        
        self.w = torch.randn((self.x.shape[1], 1), requires_grad=True)  # one weight per feature
        self.lr = 0.01
        self.final_round = 0
        logging.info(f"Initilizing node {NODE_ID} with shape: {self.x.shape} and  weights :{self.w.shape} /// {self.x} /// {self.w}")

    def get_parameters(self, config):
        logger.info(f"(get function) node {NODE_ID} , the sent weights are : {[self.w.detach().numpy()]}")
        return [self.w.detach().numpy()]

    def fit(self, parameters, config):
        z = self.x @ self.w  # [n_samples, num_features] x [num_features, 1] = [n_samples, 1]
        logger.info(f"(fit function) node {NODE_ID}, the sent (z=x*w) is : {[z.detach().numpy()]}")
        return [z.detach().numpy()], len(self.x), {"node_id": NODE_ID}


    def evaluate(self, parameters, config):
        self.final_round+=1
        dz = parameters[int(NODE_ID)]  # Gradient of loss w.r.t. z
        if np.all(dz == 0) :
            raise ValueError(f"the node {NODE_ID} received a none value as gradients.")
        
        logger.info(f"(evaluation function) node {NODE_ID} the receided parameters are : {parameters}")
        logger.info(f"(evaluation function) node {NODE_ID}, the received gradients are : {dz}")
        
        dz = torch.tensor(dz, dtype=torch.float32)
        
        z = self.x @ self.w
        z.backward(dz)
        
        logger.info(f"(evaluation function) the last weights for node {NODE_ID} are : {self.w.shape} /// {self.w}")

        with torch.no_grad():
            self.w -= self.lr * self.w.grad
        self.w.grad.zero_()

        logger.info(f"(evaluation function) the updated weights for node {NODE_ID} are : {self.w.shape} {self.w}")

        if self.final_round == 30 :
            functions.save_weights(self,NODE_ID,'ml_c')

        return 0.0, len(self.x), {}

if __name__ == "__main__":

    features = ["gender","marital_status","valuenum","los"]
    data_path = "../data/data.csv"

    private_key = Path(f"../auth_keys/node{NODE_ID}_key")
    public_key = Path(f"../auth_keys/node{NODE_ID}_key.pub")
    ca_cert = Path(f"../certs/ca.pem").read_bytes()

    fl.client.start_numpy_client(server_address="v_central_server:5000", client=VFLClient(data_path,features),
        root_certificates=ca_cert,
        insecure=False,)
      # authentication_keys=(private_key, public_key),) authentication_keys are not supported in the default gRPC+TLS transport, This feature (authentication_keys) only works with the experimental HTTP/2-based transport layer
    



