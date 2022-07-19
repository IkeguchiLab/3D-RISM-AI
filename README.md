# 3D-RISM-AI

Supporting Information of 3D-RISM-AI

## Data files
**train_data.csv**: train data for 3D-RISM-AI<br>
**test_data.csv**:  test data for 3D-RISM-AI

## Installation
You can create the python environment for 3D-RISM-AI using

```
conda env create -f ./env.yaml -n 3D-RISM-AI
conda activate 3D-RISM-AI
```

## Usage
Simply, 
```
python Scripts/3D-RISM-AI 
```
uses XGBR as a model and ComPrtLigBind as a descriptor.

Output is regression.pdf displaying predictions of binding free energies.

The model and descriptor can be seleted using options.

```
usage: 3D-RISM-AI.py [-h] [--model {XGBR,RFR,SVR,RR}]
                     [--desc {Bind,ComBind,ComLigBind,ComPrtBind,ComPrtLigBind,LigBind,PrtBind,PrtLigBind}]

3D-RISM-AI: Machine Learning Model using 3D-RISM descriptors

optional arguments:
  -h, --help            show this help message and exit
  --model {XGBR,RFR,SVR,RR}
                        select one of models
  --desc {Bind,ComBind,ComLigBind,ComPrtBind,ComPrtLigBind,LigBind,PrtBind,PrtLigBind}
                        select one of descriptors
```


Simply, 
```
python Scripts/3D-RISM-AI 
```
uses XGBR as a model and ComPrtLigBind as a descriptor.

Output is regression.pdf displaying predictions of binding free energies.

