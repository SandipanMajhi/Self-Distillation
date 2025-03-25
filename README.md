# Self-Distillation

## Installation Instructions

Required : 
```
    python=3.12
    torch=2.4.0 cuda=12.4 
```

Installation : 
```
    > virtualenv self_distil
    > source self_distil/bin/activate
    > pip install -r requirements.txt
    > bash package_install.sh
```

## Run 
GNN model : 
```
    > python3 main.py
```

Self Distillation : 
```
    > python3 main_distil.py
```

### Results:
CORA Dataset
| Model                                    | Test Accuracy |
|------------------------------------------|---------------|
| 2 layer GCN                              | 0.801         |
| MLP Based Self Distillation Model        | 0.84399       |


