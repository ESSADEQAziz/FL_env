import flwr as fl
import os
import torch
import functions
import logging
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
    def __init__(self, target_table,target_features, device="cpu"):
        self.data=functions.preprocess_node_data_NN(target_table,target_features,"regression")
        self.data = functions.insure_none(self.data)
        self.embedding_size = self.data.shape[1]
        self.encoder = functions.ClientEncoder(input_dim=self.embedding_size)
        logger.info(f"the input dimention for the node {NODE_ID} is : {self.embedding_size}")
        self.device = device
        self.data = self.data.to(self.device)
        self.encoder = self.encoder.to(self.device)

        
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.01)
        self.latest_embedding = None

        logger.info(f"Initializing the client class within node {NODE_ID} , with features {self.data.shape} .")
        
    def get_parameters(self, config):
        logger.info(f"(get function) node {NODE_ID}.")
        return []
    

    def fit(self, parameters, config):
        logger.info(f"(fit function) node {NODE_ID} , the received parameters are : {parameters} ")
        embeddings = self.encoder(self.data)
        logger.info(f"the sent embedding from node {NODE_ID} is : {[embeddings.detach().numpy()]}")
        return [embeddings.detach().numpy()], len(self.data), {"node_id": NODE_ID} 
    
    
    def evaluate(self, parameters, config):
        logger.info(f"(evaluate) node {NODE_ID} , the received parameters are : {parameters}")
        grad_np = parameters[int(NODE_ID)]
        if np.all(grad_np == 0) :
            raise ValueError(f"the node {NODE_ID} received a none value as gradients.")

        grad = torch.tensor(grad_np, dtype=torch.float32).to(self.device)
        embeddings = self.encoder(self.data)

        embeddings.backward(grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return 0.0, len(self.data), {}
    

if __name__ == "__main__" :

    # before testing , make sure that the numbers of sum(embeddings) from nodes == the input dimention of the global model within the server (because it may be some categorical features can scale dimention due to one hot encoder)
    target_features=["race","gender","anchor_age"]
    target_table = "../data/data.csv"


    fl.client.start_numpy_client(server_address="v_central_server:5000", client=VFLClient(target_table,target_features))

