import logging
import pandas as pd
import numpy as np
import json
import os
import flwr as fl
from imputation import functions

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

   
def get_client(feature,path):
    """Get the appropriate client based on the imputation strategy."""
    strategy = os.environ.get("IMPUTATION_STRATEGY", "statistical")
    
    if strategy == "statistical":
        from imputation.statistical import NodeClient
        return NodeClient(feature,path)
    elif strategy == "machine_learning":
        from imputation.machine_learning import NodeClient
        return NodeClient()
    elif strategy == "deep_learning":
        from imputation.deep_learning import NodeClient
        return NodeClient()
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

if __name__ == "__main__":
    path_to_missing_data = functions.create_missing_values('../data/chartevents.csv','valuenum',0.1)
    client = get_client('valuenum',path_to_missing_data)
    fl.client.start_numpy_client(
        server_address="central_server:5000",
        client=client
    )