`windows_tabpfn.py`: Main code, including the running and evaluation of eight models using only cohort data, cohort + wearable models, etc., as well as external validation. The top features and their corresponding data have been saved and can be read directly.

`stat_fea.py`: Code for feature engineering extraction from accelerometer data, generating 65 derived features.

`windows_tabpfn.py`: Running, evaluating, and externally validating the eight models using only cohort data, cohort + wearable models, etc., in three time windows (0-2, 0-4, 0-6).

`windows test.py`: Performs AUC difference tests on different models between each window to distinguish whether there are indeed statistically significant differences in AUC values ​​between different windows.

`model_tabpfn.py`: The Tabfn model was used to classify dementia, trained using data from England and geographically validated using data from Scotland and Wales.

Required Python version: `Python==3.10.20`

Please run the following command to replicate the environment:

`conda create -n dementia python=3.10`

`conda activate dementia`

`pip install -r requirements.txt`

`Randomly generated data/importance_tabpfn.csv:`
It contains the top 20 most important features and their specific values.

`Randomly generated data/sample_data.csv:`
UKB's sample anonymized data is randomly generated; please replace it with your actual data when running the program.