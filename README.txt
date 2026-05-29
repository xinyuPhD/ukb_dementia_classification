`windows_tabpfn`: Main code, including the running and evaluation of eight models using only queued data, queued + wearable models, etc., as well as external validation. The top features and their corresponding data have been saved and can be read directly.

`stat_fea.py`: Code for feature engineering extraction from accelerometer data, generating 65 derived features.

`windows_tabpfn`: Running, evaluating, and externally validating the eight models using only queued data, queued + wearable models, etc., in three time windows (0-2, 0-4, 0-6).

`windows test.py`: Performs AUC difference tests on different models between each window to distinguish whether there are indeed statistically significant differences in AUC values ​​between different windows.

`model_tabpfn.py`: The Tabfn model was used to classify dementia, trained using data from England and geographically validated using data from Scotland and Wales.
