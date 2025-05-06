import logging
import os 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/central_server.log"),
    ]
)
logger = logging.getLogger("central_server")


if __name__ == "__main__":
    # Get imputation strategy from environment
    strategy = os.environ.get("IMPUTATION_STRATEGY", "statistical")
    if strategy == "statistical":
        from imputation.statistical import start_server
        start_server()
    elif strategy == "machine_learning":
        from server.imputation.machine_learning_regression import start_server
        start_server()
    elif strategy == "deep_learning":
        from server.imputation.deep_learning_regression import start_server
        start_server()
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")
