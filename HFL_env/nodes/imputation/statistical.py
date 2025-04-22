import logging
import os
from pathlib import Path
import flwr as fl
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
logger.info(f"Starting node {NODE_ID}")


# Define a Flower client
class NodeClient(fl.client.NumPyClient):
    def __init__(self, target_table, target_feature, path_to_missing_data):
        self.node_id = NODE_ID
        self.path_to_missing_data=path_to_missing_data
        # i put the initialisation within the _init_ because the server will choose one node randomly for initialisation before others within get_parameters 
        self.local_parameters=functions.calculate_statistics(target_table, target_feature)
        
    def get_parameters(self,config):
        logger.info(f"inside the get function for the node {self.node_id}, No need to initilise with any parameters (we just sent those parameters just to run the process). ")
        return []

    def fit(self, parameters, config):
        logger.info(f'inside the fit function for the node {self.node_id}, the sent parameters are : {self.local_parameters}')
        return self.local_parameters, len(parameters), {}

    def evaluate(self, parameters, config):
        
        functions.evaluate_statistical_values(self.path_to_missing_data,self.local_parameters,parameters,NODE_ID)
        logger.info(f'We are inside the evaluate function within the node {NODE_ID}, the metrics saved successfully .')
        return 0.0, len(parameters), {}

    

if __name__ == "__main__":

    target_table='../data/icustays.csv'
    target_feature='los'
    missing_rate = 0.1

    
    private_key = Path(f"../auth_keys/node{NODE_ID}_key")
    public_key = Path(f"../auth_keys/node{NODE_ID}_key.pub")
    ca_cert = Path(f"../certs/ca.pem").read_bytes()

    path_to_missing_data = functions.create_missing_values(target_table,target_feature,missing_rate,NODE_ID)
    client = NodeClient(target_table,target_feature,path_to_missing_data).to_client()
    fl.client.start_client(server_address="central_server:5000", client=client,
        root_certificates=ca_cert,
        insecure=False,)
        # authentication_keys=(private_key, public_key),) authentication_keys are not supported in the default gRPC+TLS transport, This feature (authentication_keys) only works with the experimental HTTP/2-based transport layer
    

