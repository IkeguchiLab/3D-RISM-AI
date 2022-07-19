#!/usr/bin/env python

# import packages
import sys
import os
work_dir = '.'
script_path = f'{work_dir}/Scripts'
sys.path.append(script_path)
from FileTools import read_inf_file, create_model
from ReadDataFiles import read_data_files, clean_data, wide_display

# parse arguments
import argparse
parser = argparse.ArgumentParser(description='3D-RISM-AI: Machine Learning Model using 3D-RISM descriptors')
parser.add_argument('--model', choices=['XGBR','RFR','SVR','RR'], default='XGBR', help='select one of models')
parser.add_argument('--desc', choices=['Bind','ComBind','ComLigBind','ComPrtBind','ComPrtLigBind','LigBind','PrtBind','PrtLigBind'], default='ComPrtLigBind',help='select one of descriptors')
args=parser.parse_args()

# select descriptor and model
descriptor = args.desc
modelname = args.model
print(f'model: {modelname}')
print(f'descriptor: {descriptor}')
inf_file = f'{work_dir}/Models/{modelname}/{descriptor}/Inf.txt'
descriptor_file = f'{work_dir}/Descriptors/{descriptor}.txt'
matrix_data_dir = f'{work_dir}/out_matrix/'

# create and edit matrix data
data_files = [matrix_data_dir + f for f in sorted(os.listdir(matrix_data_dir))]
all_data = read_data_files(data_files)
all_data = clean_data(all_data)

# read Inf.txt
Model, params = read_inf_file(inf_file)
MainData, SubData = read_inf_file(descriptor_file)
model = create_model(Model, params)
t_list = MainData + SubData
print(model)

# Preparation of learning
from RunData import TrainTestData
run_data = TrainTestData(all_data, t_list, work_dir=work_dir)

# learning and prediction
if modelname == 'XGBR':
  run_data.train_model(model)
  train_li, test_li = run_data.get_test_score(model)
  print(train_li, test_li)
  evals_result = model.evals_result()
else:
  model.fit(run_data.train_t, run_data.train_a)
  train_scores, test_scores = run_data.get_test_score(model)
  print(train_scores, test_scores)


# calculate difference between predictions and experimental values
gap_data = run_data.get_gap_values(all_data)
print(gap_data.shape)
#Low Absolute Error
HAE_data = gap_data[gap_data['gap_values'] > 2.0]
print(HAE_data.shape)
#High Absolute Error
LAE_data = gap_data[gap_data['gap_values'] <= 2.0]
print(LAE_data.shape)

# settings for plot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['mathtext.default'] = 'default'
plt.rcParams['figure.dpi'] = 600
plt.rcParams["font.size"] = 12

def plot_scatter(df_list,  modelname, x_title, y_title, fname=None):
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    plt.plot(range(-18, 0), range(-18, 0), linestyle='dashed', color='black', linewidth=1, zorder=3)
    scats = []
    for df in df_list:
        df_x = df[x_title]
        df_y = df[y_title]
        scat = ax.scatter(df_x, df_y, s=3, zorder=2)
        scats.append(scat)
    df_x = df_list[0][x_title]
    df_y = df_list[0][y_title]
    peason_r = df_x.corr(df_y)
    spearman_rho = df_x.corr(df_y, method='spearman')
    rmse = np.sqrt(mean_squared_error(df_x.T, df_y.T))
    plt.title(f'{modelname}   R: {peason_r:.2f} rho: {spearman_rho:.2f} RMSE: {rmse:.2f}')
    plt.xticks(range(-16, -1, 2), fontsize=10)
    plt.yticks(range(-16, -1, 2), fontsize=10)
    plt.xlim(-17, -1)
    plt.ylim(-17, -1)
    plt.xlabel('exp dG (kcal/mol)')
    plt.ylabel('pred dG (kcal/mol)')
    #ax.legend(handles=scats, labels=[r'AE $\leq$ 2', r'AE $>$ 2'], fontsize=10, markerscale=2, handlelength=1, loc='upper left')
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight", pad_inches=0.1)

# plot regression
plot_scatter([gap_data, HAE_data], modelname, 'G_experiment', 'test_predict', 'regression.pdf')
