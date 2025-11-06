This is for practical applications of forecasting IMTS data in clinic, physics, telecom, security, biology, finance etc.

Installation:
Please run 'pip -r install requirements.txt'

Experiment:
The complete command is:
Example:
```shell
python run.py 
    --data_dir {data_dir} \
    --graph_dir {graph_dir} \
    --log_dir {log_dir} \
    --output {output_dir} \
```

- `data_dir`: type=str,default='./data/', 'directory of the original data.'  
- `graph_dir`,type=str,default='./graph/', 'directory of graphs.'
- `output_dir`,type=str,default='./output/', 'directory of outputs.'
- `log_dir`,type=str,default='./log/', 'directory of the transaction logs.'


The configuration file is in `./model/Config.py`, you can change the parameters there.

To get the forecasting result,
please run:
```shell
'python run.py '
```
or
```shell
'python run.py --data_dir ./data/eeg/eeg.csv'
```
