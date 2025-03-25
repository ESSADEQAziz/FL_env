import flwr as fl

fl.server.start_server(server_address="central_server:5000",
                       config=fl.server.ServerConfig(num_rounds=2))