import logging
import os
import flwr as fl


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
    def __init__(self):
        self.node_id = NODE_ID

        
  