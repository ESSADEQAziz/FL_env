import logging
import os
import flwr as fl
import numpy as np
from pathlib import Path
import functions
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

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
    def __init__(self, target_table, feature_x, feature_y,missing_rate):
        self.epochs = 100
        self.iterations=100
        self.server_round=0
        
        # Load and preprocess data
        X, Y = functions.load_and_preprocess_data(target_table, feature_x, feature_y)
        logger.info(f"Loaded {len(X)} samples from {target_table} using features '{feature_x}' and '{feature_y}'")
        logger.info(f"The size of the features is: '{X.size}' and '{Y.size}'")
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = functions.split_reshape_normalize(X, Y, test_size=missing_rate, random_state=42)

        # Initialize model with warm start for partial_fit
        # By default SGDRegressor work with mini-batch 'small sample of data'
        self.model = SGDRegressor(
            warm_start=True, 
            max_iter=self.iterations, 
            tol=1e-3, 
            learning_rate="adaptive", 
            eta0=0.001
        )
        
    def fit(self, parameters, config):

        # Update model parameters
        self.model.coef_ = np.array(parameters[0]).reshape(1, -1)
        self.model.intercept_ = np.array(parameters[1])

        # Reshape coefficients for partial_fit
        self.model.coef_ = self.model.coef_.reshape(-1)
        self.model.intercept_ = self.model.intercept_.reshape(-1)

        # Train the model with the partial_fit method does not apply epochs automatically, we need to do it manually.
        for _ in range(self.epochs):
            self.model.partial_fit(self.X_train, self.Y_train.ravel()) 


        self.server_round = config.get("server_round", -1)

        logger.info(f"the sent parameters from node {NODE_ID} are (round = {self.server_round }) : a= {self.model.coef_}  b= {self.model.intercept_}")
        logger.warning(f" (fit function) the server round = {self.server_round } ")

        return [self.model.coef_, self.model.intercept_], len(self.X_train), {}
    

    def get_parameters(self, config):
        """Return the model parameters."""
        logger.info(f"Node {NODE_ID} get_parameters: coef={0.0}, intercept={0.0}")
        return np.array([0.0, 0.0])

    def evaluate(self, parameters, config):
        """Evaluate model performance."""

        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

            # Make predictions
        y_pred=self.model.predict(self.X_test)
        mse = mean_squared_error(self.Y_test , y_pred)
        
        logger.info(f" (evaluation function ) Node {NODE_ID} make predictions with parameters: coef={parameters[0]}, intercept={parameters[1]}")
        logger.info(f"The Calculated MSE for node  {NODE_ID}: MSE={mse:.4f}")


        logger.error(f" node {NODE_ID} parameters {parameters} mse {mse} server_rd {self.server_round}")
        functions.evaluate_ml_values(NODE_ID,parameters,mse,self.server_round)
        
        return mse, len(self.X_test),{}
    
if __name__ == "__main__":
    target_table = "../data/labevents.csv"
    missing_rate = 0.1
    feature_x = "value"
    feature_y = "valuenum"


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
    