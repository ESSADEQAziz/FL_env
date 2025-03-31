import logging
import flwr as fl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/central_server.log"),
    ]
)
logger = logging.getLogger("central_server")

def start_server():

    # Start the federated learning server 
    fl.server.start_server(
        server_address="central_server:5000",
        config=fl.server.ServerConfig(num_rounds=2),
        #strategy=strategy,
    )

if __name__ == "__main__":
    # Start Flower client
    start_server()