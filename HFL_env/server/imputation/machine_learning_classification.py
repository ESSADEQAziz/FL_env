import logging
import flwr as fl
import functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/central_server.log"),
    ]
)
logger = logging.getLogger("central_server")
logger.info("Starting the central server ...")

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round, results, failures):
        accuracy=[]
        if failures:
            logger.warning(f"Failures: {len(failures)} clients failed to send updates")
            raise ValueError(f"Failure : {failures}")

        weights_list = []
        biases_list = []
        total_samples = 0

        for client, fit_res in results:
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            num_samples = fit_res.num_examples
            input_dim = fit_res.metrics.get("input_dim")
            node_id = fit_res.metrics.get("node_id")
            output_dim = fit_res.metrics.get("output_dim")
            accuracy.append(fit_res.metrics.get("accuracy"))

            logger.info(f"Samples: {num_samples} /node: {node_id}, Input Dim: {input_dim}, Output_dim: {output_dim}, Params: {params}")

            if len(params) != 2:
                logger.error(f"Unexpected parameter format: {params}")
                continue

            weight, bias = params
            weights_list.append((weight, num_samples))
            biases_list.append((bias, num_samples))
            total_samples += num_samples

        logger.info(f"Weight entries: {len(weights_list)}, Bias entries: {len(biases_list)}, Total Samples: {total_samples}")

        if total_samples == 0:
            logger.error("No valid data received from nodes.")
            return None, {}

        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        metrics["accuracy"] = sum(accuracy)/len(accuracy)
        logger.info(f"Aggregated Parameters: {aggregated_parameters}, Metrics: {metrics}")

        if server_round == 20:
            functions.save_model(aggregated_parameters, server_round, input_dim, "ml_c",num_classes=output_dim)
            logger.info("The model saved successfully.")

        return aggregated_parameters, metrics

    def evaluate(self, server_round, parameters):
        # Optional: centralized evaluation logic can be added here
        return 0.0, {'accuracy' : 0.0}

def start_server():
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    history = fl.server.start_server(
        server_address="central_server:5000",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=20) #if you change the num_round, change it also within the save_dl_model() function
    )
    return history

if __name__ == "__main__":
    functions.save_metrics(start_server(), "ml_c")
