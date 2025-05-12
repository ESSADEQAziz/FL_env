import functions

x,_,_ = functions.prepare_mimic_data("./data/node1","lab_results")
x.to_csv('my_data.csv', index=False)