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
python Scripts/3D-RISM-AI.py 
```
predicts binding free energies using XGBR as a model and ComPrtLigBind as a descriptor.

Outputs are predictions.csv containing predicted binding free energies and regression.pdf displaying predictions of binding free energies.

You can use other models and descriptors using options.

```
usage: 3D-RISM-AI.py [-h] [--model {XGBR,RFR,SVR,RR}]
                     [--desc {Bind,ComBind,ComLigBind,ComPrtBind,ComPrtLigBind,LigBind,PrtBind,PrtLigBind}]
                     [--out OUT] [--fig FIG]
                     [--train TRAIN] [--test TEST]

3D-RISM-AI: Machine Learning Model using 3D-RISM descriptors

optional arguments:
  -h, --help            show this help message and exit
  --model {XGBR,RFR,SVR,RR}
                        select one of models (default: XGBR)
  --desc {Bind,ComBind,ComLigBind,ComPrtBind,ComPrtLigBind,LigBind,PrtBind,PrtLigBind}
                        select one of descriptors (default:ComPrtLigBind)
  --out OUT             output file name (default: predictions.csv)
  --fig FIG             output figure file name (default: regression.pdf)
  --train TRAIN         input train data file name (default: train_data.csv)
  --test TEST           input test data file name (default: test_data.csv)
```

## Citation

For details of 3D-RISM-AI, see the following paper.
If our models or any scripts are useful to you, consider citing the following paper in your publications:

Kazu Osaki, Toru Ekimoto, Tsutomu Yamane, and Mitsunori Ikeguchi, 3D-RISM-AI: A machine learning approach to predict protein-ligand binding affinity using 3D-RISM, J. Phys. Chem. B in press.


