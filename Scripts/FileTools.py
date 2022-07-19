import os
import sys

def conf_dir(file_path, act):
    if os.path.exists(file_path) == False:
        if act:
            os.makedirs(file_path)
        else:
            sys.exit(f'{file_path} not found')

def read_inf_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    info_list = []
    for line in lines:
        line = line.strip('\n')
        if line.startswith('Model'): info_list.append(line.split()[1])
        elif line.startswith('Params'): info_list.append({})
        elif line.startswith('GSCPU'): info_list.append(int(line.split()[1]))
        elif line.startswith('MainData'): info_list.append(line.split()[1:])
        elif line.startswith('SubData'): info_list.append(line.split()[1:])
        elif line.startswith('OutFile'): info_list.append(line.split()[1])
        elif line.startswith('Descriptor'): info_list.append(line.split()[1])
        elif line.startswith('metric'): info_list.append(line.split()[1:])
        elif line.startswith('No'): info_list.append(line.split()[1])
        elif line.startswith('#'): continue
        else:
            data = line.split()
            key = data[0]
            items = []
            if key == 'hidden_layer_sizes':
                for i in data[1:]:
                   if ',' in i: items.append(tuple(map(int, i.split(','))))
                   else: items.append((int(i),))
            else:
                try:
                    for i in data[1:]:
                       if '.' in i: items.append(float(i))
                       else: items.append(int(i)) 
                except:
                    items.append(i)
            info_list[1][key] = items
    return info_list

def create_model(Model, params=None):
    if params: 
        for key, item in params.items():
            params[key] = item[0]
    if Model == 'RFR':
        from sklearn.ensemble import RandomForestRegressor as RFR
        if params: model = RFR(**params)
        else: model = RFR()
    elif Model == 'SVR':
        from sklearn.svm import SVR
        if params: model = SVR(**params)
        else: model = SVR()
    elif Model == 'GBR':
        from sklearn.ensemble import GradientBoostingRegressor as GBR
        if params: model = GBR(**params)
        else: model = GBR()
    elif Model == 'XGBR':
        from xgboost import XGBRegressor as XGBR
        model = XGBR()
    elif Model == 'LGBR':
        from lightgbm import LGBMRegressor as LGBM
        model = LGBM()
    elif Model == 'ABR':
        from sklearn.ensemble import AdaBoostRegressor as ABR
        model = ABR()
    elif Model == 'MLPR':
        from sklearn.neural_network import MLPRegressor as MLPR
        if params: model = MLPR(**params)
        else: model = MLPR()
    elif Model == 'RR':
        from sklearn.linear_model import Ridge as RR
        if params: model = RR(**params)
        else: model = RR()
    else:
        sys.exit(f'{Model} not set')
    if params: model.set_params(**params)
    return model
