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
        logger.info(f"inside the get function for the node {self.node_id} the local parameters are : mean {self.local_parameters} ")
        return self.local_parameters

    def fit(self, parameters, config):
        logger.info(f'inside the fit function for the node {self.node_id} : the aggregated parameters are : {parameters}')
        return self.local_parameters, len(parameters), {}

    def evaluate(self, parameters, config):
        
        agg_mean = functions.evaluate_statistical_values(self.path_to_missing_data,self.local_parameters,parameters,NODE_ID)
        logger.info(f'We are inside the evaluate function within the node {NODE_ID} and the agg_mean is : {agg_mean}')
        return agg_mean, len(parameters), {}

    

if __name__ == "__main__":

    target_table='../data/icustays.csv'
    target_feature='los'
    missing_rate = 0.1

    # Load certificates 
    # ca_cert, private_key, public_key = functions.process_cert_key(NODE_ID)
    
    # private_key = Path(f"../certs/node{NODE_ID}.key").read_bytes()
    # public_key = Path(f"../certs/node{NODE_ID}.pem").read_bytes()
    # ca_cert = Path(f"../certs/ca.pem").read_bytes()

    path_to_missing_data = functions.create_missing_values(target_table,target_feature,missing_rate,NODE_ID)
    client = NodeClient(target_table,target_feature,path_to_missing_data).to_client()
    fl.client.start_client(server_address="central_server:5000", client=client)
        # root_certificates=ca_cert,
        # insecure=False,
        # authentication_keys=(private_key, public_key),)
    

