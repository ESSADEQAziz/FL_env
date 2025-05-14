import logging
import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from functions import SimpleClassifier
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
    def __init__(self, target_table,feature_x,feature_y):
        self.epochs = 10
        self.iterations=10  
        self.learning_rate=0.01
        self.accuracy = 0
        self.criterion = nn.CrossEntropyLoss()

        # Load and preprocess data
        X, Y = preprocess_node_data(target_table, feature_x, feature_y,'dl_c')
        self.input_dim = X.shape[1]
        self.num_classes= Y.shape[1]
        logger.info(f"Loaded {len(X)} samples from {target_table} node {NODE_ID} using features {feature_x} and {feature_y}")
        logger.info(f"The size of the features is: '{X.shape}' and '{Y.shape}'")

        # Split data into train and test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = split_reshape_normalize(X, Y, test_size=0.2, random_state=42)

        logger.info(f"the result of test_split_reshape() : x_train = {self.X_train.shape} x_test = {self.X_test.shape} y_train = {self.Y_train.shape} y_test= {self.Y_test.shape}")
        self.model = SimpleClassifier(input_dim= self.input_dim ,num_classes= self.num_classes) 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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
        logger.info(f"( fit function ) node {NODE_ID}, the received parameters are : {parameters}")
        self.set_parameters(parameters)
        
        #Prepare data
        train_dataset = torch.utils.data.TensorDataset(
            self.X_train,
            torch.argmax(self.Y_train, dim=1)  # convert one-hot Y to class index
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        # Loss and optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        logger.info(f"(fit function ) node {NODE_ID}, the sent parameters are : {self.get_parameters(config)} with the last accuracy : {self.accuracy}")
        return self.get_parameters(config), len(self.X_train), {"input_dim": self.input_dim,"num_classes":self.num_classes, "accuracy": self.accuracy}
        
    def evaluate(self, parameters, config):
        logger.info(f"(evaluation function) node {NODE_ID} the getting parameters are : {parameters} ")
        self.set_parameters(parameters)

        Y_test = torch.argmax(self.Y_test, dim=1)  
        # Y_test = self.Y_test.to(self.device).long()
       
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test)
            loss = self.criterion(y_pred, Y_test)

            preds = torch.argmax(y_pred, dim=1)
            logger.warning(f"the shape test tensor is : {Y_test.shape}, and the shape of the predicted tensor is : {preds.shape}")
            correct = (preds == Y_test).sum().item()
            accuracy = correct / Y_test.size(0)

        self.accuracy = float(accuracy) 

        logger.info(f"the sent loss {NODE_ID}: {float(loss.item())} and the sent accuracy {NODE_ID}: {float(accuracy)}")

        return float(loss.item()), len(self.X_test), {"accuracy": float(accuracy)}

        
if __name__ == "__main__":

    target_table = "../data/extracted_lab_results.csv"
    feature_x = ['creatinine','blood_glucose']
    feature_y = "gender"


    private_key = Path(f"../auth_keys/node{NODE_ID}_key")
    public_key = Path(f"../auth_keys/node{NODE_ID}_key.pub")
    ca_cert = Path(f"../certs/ca.pem").read_bytes()
    
    
    client = NodeClient(target_table, feature_x, feature_y).to_client()
    fl.client.start_client(server_address="central_server:5000", client=client,
        root_certificates=ca_cert,
        insecure=False,)
      # authentication_keys=(private_key, public_key),) authentication_keys are not supported in the default gRPC+TLS transport, This feature (authentication_keys) only works with the experimental HTTP/2-based transport layer
    