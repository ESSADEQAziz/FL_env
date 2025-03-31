import logging
import os
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
NODE_ID = os.environ.get("NODE_ID", "1")
logger.info(f"Starting node {NODE_ID}")


# Define a Flower client
class NodeClient(fl.client.NumPyClient):
    def __init__(self, feature,path_to_missing_data):
        self.node_id = NODE_ID
        self.feature = feature
        # i put the initialisation within the _init_ because the server will choose one node randomly for initialisation before others within get_parameters 
        self.local_parameters=functions.calculate_statistics(f'../data/chartevents.csv', self.feature)
        
    def get_parameters(self,config):
        logger.info(f"inside the get function for the node {self.node_id} the local parameters are : mean {self.local_parameters} ")
        return self.local_parameters

    def fit(self, parameters, config):
        logger.info(f'inside the fit function for the node {self.node_id} : the aggregated parameters are : {parameters}')
        return self.local_parameters, len(parameters), {}

    def evaluate(self, parameters, config):
        logger.info(f'We are inside the evaluate function within the node {NODE_ID}.')
        agg_mean = functions.evaluate_values(path_to_missing_data,self.local_parameters,parameters,f'../results/metrics_node_{NODE_ID}.json')
        return agg_mean, len(parameters), {}

    

if __name__ == "__main__":
   
    path_to_missing_data = functions.create_missing_values('../data/chartevents.csv','valuenum',0.1)
    client = NodeClient('valuenum',path_to_missing_data)
    fl.client.start_numpy_client(server_address="central_server:5000", client=client)
    

