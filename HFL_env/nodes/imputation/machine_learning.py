import logging
import os
import flwr as fl
import torch.optim as optim
import torch.nn as nn
import torch
from pathlib import Path
import functions


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
logger.info(f"Starting node {NODE_ID} ...")

# Define a Flower client
class NodeClient(fl.client.NumPyClient):
    def __init__(self, target_table, feature_x, feature_y,missing_rate):
        self.epochs = 100
        self.iterations=100
        self.server_round=0
        self.learning_rate=0.01

        X, Y = functions.preprocess_node_data(target_table, feature_x, feature_y,'ml_r')
        logger.info(f"the features ({feature_x}) {X.shape} and the target  ({feature_y}) {Y.shape} laoded from table {target_table}")
        # Split data into train and test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = functions.split_reshape_normalize(X, Y, test_size=missing_rate, random_state=42)
        logger.info(f"X_train {self.X_train.shape},Y_train {self.Y_train.shape},X_test {self.X_test.shape} Y_test {self.Y_test.shape}")

        self.input_dim = X.shape[1]
        self.model = functions.LinearRegressionModel(input_dim=self.input_dim)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        
    def fit(self, parameters, config):
        logger.info(f"(fit function) node {NODE_ID}, the received parameters are : {parameters}")
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(self.epochs):  # local epochs
            preds = self.model(self.X_train)
            loss = self.loss_fn(preds, self.Y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        logger.info(f"(fit function) the sent parameters, node {NODE_ID} are : {self.get_parameters(config={})} ")
        return self.get_parameters(config={}), len(self.X_train), {"input_dim": self.input_dim}


    def get_parameters(self, config):
        weight = self.model.linear.weight.data.numpy()
        bias = self.model.linear.bias.data.numpy()
        logger.info(f"(get function) node {NODE_ID} , the sent parameteres are {[weight, bias]}")
        return [weight, bias]
    
    def set_parameters(self, parameters):
        weight, bias = parameters
        self.model.linear.weight.data = torch.tensor(weight, dtype=torch.float32)
        self.model.linear.bias.data = torch.tensor(bias, dtype=torch.float32)
        logger.info(f"(set function) node {NODE_ID}, parameters updated successfully.")

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        preds = self.model(self.X_test)
        loss = self.loss_fn(preds, self.Y_test)

        logger.info(f"(evaluate function) node {NODE_ID}, the sent loss is {float(loss.item())}")
        return float(loss.item()), len(self.X_train), {}
  
    
if __name__ == "__main__":
    target_table = "../data/labevents.csv"
    missing_rate = 0.2
    feature_x = ['valuenum','ref_range_lower','priority']
    feature_y = "ref_range_upper"


    private_key = Path(f"../auth_keys/node{NODE_ID}_key")
    public_key = Path(f"../auth_keys/node{NODE_ID}_key.pub")
    ca_cert = Path(f"../certs/ca.pem").read_bytes()
    
    """ For the machine learning approche, we don't need the create_missing_values() function,
    because the purpose of it is to perserve a data nerver seen by the model to evalute the performence,
    and we already apply the same process implecitlly using the train_test_split logic where (missing_rate = = test_size)  """

    client = NodeClient(target_table, feature_x, feature_y,missing_rate).to_client()
    fl.client.start_client(server_address="central_server:5000", client=client,
        root_certificates=ca_cert,
        insecure=False,)
      # authentication_keys=(private_key, public_key),) authentication_keys are not supported in the default gRPC+TLS transport, This feature (authentication_keys) only works with the experimental HTTP/2-based transport layer
    