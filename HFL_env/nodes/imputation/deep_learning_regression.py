import logging
import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from functions import SimpleRegressor
from functions import split_reshape_normalize
from functions import preprocess_node_data


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
    def __init__(self, target_table,feature_x,feature_y,missing_rate):
        self.epochs = 10
        self.iterations=10
        self.server_round=0   

        # Load and preprocess data
        X, Y = preprocess_node_data(target_table, feature_x, feature_y,'dl')
        self.input_dim = X.shape[1]
        logger.info(f"Loaded {len(X)} samples from {target_table} node {NODE_ID} using features {feature_x} and {feature_y}")
        logger.info(f"The size of the features is: '{X.shape}' and '{Y.shape}'")

        # Split data into train and test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = split_reshape_normalize(X, Y, test_size=missing_rate, random_state=42)

        logger.info(f"the result of test_split_reshape() : x_train = {self.X_train.shape} x_test = {self.X_test.shape} y_train = {self.Y_train.shape} y_test= {self.Y_test.shape}")
        self.model = SimpleRegressor(input_dim= self.input_dim) 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        logger.info(f"( get_parameters ) from node {NODE_ID} with a size of : {[val.cpu().numpy() for val in self.model.state_dict().values()]}")
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        logger.info(f"( set_parameters ) node {NODE_ID} ")
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)
     
    def fit(self, parameters, config):
        logger.info(f"(fit function ) node {NODE_ID}, the parameters are : {parameters}")
        self.set_parameters(parameters)
        self.model.train()

        dataset = TensorDataset(self.X_train, self.Y_train)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        for epoch in range(self.epochs):
            for x_batch, y_batch in loader:
                self.optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
        
        logger.info(f"(fit function ) node {NODE_ID}, the sent parameters are : {self.get_parameters(config)}")

        return self.get_parameters(config), len(dataset), {"input_dim": self.input_dim}
    
    def evaluate(self, parameters, config):
        logger.info(f"(evaluation function) node {NODE_ID} the getting parameters are : {parameters} ")
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test)
            loss = self.criterion(y_pred, self.Y_test)
        logger.info(f"the sent loss {NODE_ID}: {float(loss.item())}")
        return float(loss.item()), len(self.X_test), {"loss": float(loss.item())}
    
if __name__ == "__main__":
    target_table = "../data/labevents.csv"
    missing_rate = 0.2
    feature_x = ['valuenum','flag']
    feature_y = "ref_range_lower"


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
    