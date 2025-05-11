import functions


for i in range(5) :
    x,_,_ = functions.prepare_mimic_data(f'../data/node{i+1}',"vital_signs")
    x.to_csv(f'extracted_vital_signs_node{i+1}.csv', index=False)

    y,_,_ = functions.prepare_mimic_data(f'../data/node{i+1}',"lab_results")
    y.to_csv(f'extracted_lab_results_node{i+1}.csv', index=False)


