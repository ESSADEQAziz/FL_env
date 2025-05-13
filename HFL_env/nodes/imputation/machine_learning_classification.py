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
    def __init__(self, target_table, features_x, feature_y):
        self.epochs = 100
        self.learning_rate = 0.01
        self.accuracy=0

        X, Y = functions.preprocess_node_data(target_table, features_x, feature_y, 'ml_c')
        logger.info(f"the features ({features_x}) {X.shape} and the target  ({feature_y}) {Y.shape} loaded from table {target_table}")

        self.X_train, self.X_test, self.Y_train, self.Y_test = functions.split_reshape_normalize(
            X, Y, test_size=0.2, random_state=42)
        logger.info(f"the x_train = {self.X_train.shape} y_train = {self.Y_train.shape} x_test = {self.X_test.shape} y_test = {self.Y_test.shape}")
        self.input_dim = X.shape[1]
        self.num_classes= Y.shape[1]
        if self.num_classes == 2 :
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.num_classes > 2 : 
            self.loss_fn = nn.CrossEntropyLoss()
        else :
            raise ValueError(f"The dimention of the target feature is not compatible.")
        
        self.model = functions.LogisticRegressionModel(input_dim=self.input_dim , output_dim= self.num_classes)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def fit(self, parameters, config):
        logger.info(f"(fit function) node {NODE_ID}, the received parameters are : {parameters}")
        self.set_parameters(parameters)
        self.model.train()
        logger.warning(f" node {NODE_ID} the shape of the x_train : {self.X_train.shape} , and y_train :{self.Y_train.shape}, and after argmax : {torch.argmax(self.Y_train, dim=1).shape} ")
            
        for epoch in range(self.epochs):
            preds = self.model(self.X_train).squeeze()
            
            if self.num_classes == 2 :
                loss = self.loss_fn(preds, self.Y_train.float())
            else : 
                loss = self.loss_fn(preds, torch.argmax(self.Y_train, dim=1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        logger.info(f"(fit function) node {NODE_ID} the sent parameters are : {self.get_parameters(config={})} ")
        return self.get_parameters(config={}), len(self.X_train), {"input_dim": self.input_dim,"node_id": NODE_ID, "output_dim" : self.num_classes, "accuracy" : self.accuracy}

    def get_parameters(self, config):
        weight = self.model.linear.weight.data.numpy()
        bias = self.model.linear.bias.data.numpy()
        logger.info(f"(get function) node {NODE_ID}, the sent parameters are {[weight, bias]}")
        return [weight, bias]

    def set_parameters(self, parameters):
        weight, bias = parameters
        self.model.linear.weight.data = torch.tensor(weight, dtype=torch.float32)
        self.model.linear.bias.data = torch.tensor(bias, dtype=torch.float32)
        logger.info(f"(set function) node {NODE_ID}, parameters updated successfully.")

    def evaluate(self, parameters, config):
        logger.info(f"(evaluate function) node {NODE_ID} the received parameters are {parameters}")
        
        self.set_parameters(parameters)
        self.model.eval()

        with torch.no_grad():
            preds = self.model(self.X_test).squeeze()

            loss = self.loss_fn(preds, self.Y_test.float())
            logger.info(f"(evaluate function) node {NODE_ID}, the sent loss is {float(loss.item())}")

            # Compute accuracy if classification task
            preds_classes = torch.argmax(preds, dim=1)
            target_classes = torch.argmax(self.Y_test, dim=1)
            correct = (preds_classes == target_classes).sum().item()
            total = target_classes.size(0)
            accuracy = correct / total if total > 0 else 0.0

            self.accuracy=accuracy

            logger.info(f"(evaluate function) node {NODE_ID}, accuracy: {accuracy:.4f}")

        return float(loss.item()), len(self.X_train), {}

if __name__ == "__main__":

    target_table = "../data/admissions.csv"
    features_x = ['insurance', 'marital_status']
    feature_y = "race"

    private_key = Path(f"../auth_keys/node{NODE_ID}_key")
    public_key = Path(f"../auth_keys/node{NODE_ID}_key.pub")
    ca_cert = Path(f"../certs/ca.pem").read_bytes()

    client = NodeClient(target_table, features_x, feature_y).to_client()
    fl.client.start_client(
        server_address="central_server:5000",
        client=client,
        root_certificates=ca_cert,
        insecure=False,)
      # authentication_keys=(private_key, public_key),) authentication_keys are not supported in the default gRPC+TLS transport, This feature (authentication_keys) only works with the experimental HTTP/2-based transport layer
    
