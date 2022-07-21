import pandas as pd
import numpy as np
import math

def miss_data(df):
    null_val = df.isnull().sum()
    miss_table = pd.concat([null_val], axis=1)
    miss_table = miss_table.rename(columns = {0: 'MissNum'})
    miss_table = miss_table[miss_table['MissNum'] != 0]
    return(miss_table.T)

def wide_display(df):
    max_column = df.shape[1]
    pd.set_option('display.max_columns', max_column)
    
def read_tsv(file_name):
    all_data = pd.read_csv(file_name, delimiter='\t', dtype='object')
    #all_data = pd.read_csv(open(file_name, 'rU'), delimiter='\t', dtype='object')
    #rows = csv.reader(io.StringI)(file_name
    all_data = all_data.set_index('Index')
    return all_data
        
def del_notrun_numrota(df, verbose):
    if verbose == 2:
        show_notrun_data(df, 'NumRota', 'not run')
        show_notrun_data(df, 'NumRota', 'False')
    df = df[df['NumRota'].str.contains('not run') == False]
    df = df[df['NumRota'] != 'False']
    return df

def show_notrun_data(df, row_name, contain_name):
    #df_nan = df[df[row_name].isnull()]
    #wide_display(df_nan)
    #display(df_nan)
    df = df[df[row_name].str.contains(contain_name)]
    index_list = " ".join(df.index.values)
    print(f'{row_name} {contain_name}:\t{df.shape[0]}\t{index_list}')

def del_notrun(df, verbose):
    not_run_contain_name_list = ['schrodinger', 'cap', 'antechamber', 'tleap', 'minimization']
    if verbose == 2:
        for not_run_contain_name in not_run_contain_name_list:
            show_notrun_data(df, 'Chain', not_run_contain_name)
    df = df[df['Chain'].str.contains('not run') == False]
    return df
    
def check_rism_run(df, verbose):
    #display(df.loc[['0424'],:])
    sasa_list = [f'{name}_SASA' for name in ['Sys', 'Prt', 'Lig']]
    not_run_row_name_list = sasa_list + ['Sys_AllPotn', 'Prt_AllPotn', 'Lig_AllPotn']
    for not_run_row_name in not_run_row_name_list:
        df_nan = df[df[not_run_row_name].str.contains('not run')]
        index_list = " ".join(df_nan.index.values)
        if verbose == 2:
            print(f'not run {not_run_row_name}:\t{df_nan.shape[0]}\t{index_list}')
        df = df[df[not_run_row_name].str.contains('not run') == False]
    return df

def del_all_nan(df, del_title, verbose):
    df_nan = df[df[del_title].isnull() != False]
    df = df[df[del_title].isnull() != True]
    index_list = " ".join(df_nan.index.values)
    if verbose == 2:
        print(f'del nan {del_title}: {df_nan.shape[0]}\t{index_list}')
    return df

def replace(df):
    replace_list = ['His','Insert','MissId', 'PrtRmsd']
    df[replace_list] = df[replace_list].replace('False', 0).replace('True', 1)
    return df

def as_int(df):
    #int_list = ['His','ChainNo','Insert','MissId','NumCA','Prt+','Lig+', 'NumRota']
    int_list = ['PrtCharge', 'LigCharge', 'HIE', 'HID', 'HIP', 'NumAmino', 'NumACE', 'NumNME', 'NumPrtRes', 'NumLigAtom', 'NumLigAtomNoh', 'NumPocketRes', 'NumRota']
    df[int_list] = df[int_list].astype(int)
    return df

def del_all_False(df, string, verbose):
    num_pre_row = df.shape[0]
    df = df.query(string)
    num_now_row = df.shape[0]
    if verbose >= 1:
        print(f'Rism False:\t{num_pre_row - num_now_row}')
    return df


def as_float(df, verbose):
    float_list = ['Resolut','pK']
    rmsd_list = ['LigNoh', 'Lig', 'PocketNoh', 'Pocket', 'CompPocketNoh', 'CompPocket']
    float_list  += [f'{name}Rmsd' for name in rmsd_list]
    sys_prt_lig_list = [ '_AllPotn', '_ChemPotn', '_SolPotn', '_Entropy', '_PartMolV', '_Solvent', '_SASA']
    pol_apol_list = ['Chem', 'SolPotn', 'Ent', 'MolV', 'Solvent']
    for name in pol_apol_list:
        pol_name = f'_Pol{name}'
        apol_name = f'_Apol{name}'
        sys_prt_lig_list += [pol_name, apol_name]
    name1_list = ['Sys', 'Prt', 'Lig']
    for name1 in name1_list:
        for name2 in sys_prt_lig_list:
            float_list.append(f'{name1}{name2}')
    if verbose == 2:
        for name in float_list:
            show_notrun_data(df, name, 'not run')
    df[float_list] = df[float_list].astype(float)
    return df
    
def tf_pep(df):
    df['LigName'] = np.where(df['LigName'].str.len() ==3, 0, 1)
    return df

def make_dummies(df):
    #df['Type'] = df['Type'].replace('Ki<', 'Ki')
    df = df[df['Type'] != 'Ki<']
    dummy_list = ['Type']
    df_dummies = pd.get_dummies(df[dummy_list])
    df = pd.concat([df, df_dummies], axis=1)
    return df

def start_up(file_name, del_false=True, verbose=0):
    data = read_tsv(file_name)
    data = del_notrun(data, verbose)
    data = del_all_nan(data, 'Lig_AllPotn', verbose)
    data = del_all_nan(data, 'Sys_AllPotn', verbose)
    data = check_rism_run(data, verbose)
    #data = replace(data)
    #string = "Prt_SESA != 'False' and Prt_SASA != 'False' and Prt_SESV != 'False' and Lig_SESA != 'False' and Lig_SASA != 'False' and Lig_SESV != 'False'"
    #data = del_all_False(data, string)
    data = del_notrun_numrota(data, verbose)
    data = as_int(data)
    data = make_dummies(data)
    if del_false:
        string = "Sys_AllPotn != 'False' and Prt_AllPotn != 'False' and Lig_AllPotn != 'False'"
        data = del_all_False(data, string, verbose)
        data = as_float(data, verbose)
    #data = tf_pep(data)
    wide_display(data)
    return data

def read_data_files(file_list, del_false=True, verbose=0):
    data_list = []
    for file_name in file_list:
        data = start_up(file_name, del_false, verbose)
        data_list.append(data)
    df = pd.concat(data_list)
    return df
    
def delta(df, data):
    name_df = f'delta_{data}'
    df_new = df[f'Sys_{data}']-(df[f'Prt_{data}']+df[f'Lig_{data}'])
    df_new = pd.Series(df_new, name=name_df)
    return df_new

def clean_data(df):
    df = df[df.index != '2873']
    #G_experiment = pd.Series(df['pK']*(-0.593)*(1/(math.log10(math.e))), name='G_experiment')
    G_experiment = pd.Series(df['pK']*(-0.593)*math.log(10), name='G_experiment')
    #G_experiment = pd.Series(df['pK']*(-0.593), name='G_experiment')
    #prt_charge2 = pd.Series(df['Prt+'].abs(), name='Prt++')
    #lig_charge2 = pd.Series(df['Lig+'].abs(), name='Lig++')
    #sys_charge = df['Prt+']+df['Lig+']
    #sys_charge = pd.Series(sys_charge, name='Sys+')
    #sys_charge2 = pd.Series(sys_charge.abs(), name='Sys++')
    G_theory = df['Sys_AllPotn']+df['Sys_ChemPotn']-(df['Prt_AllPotn']+df['Prt_ChemPotn']+df['Lig_AllPotn']+df['Lig_ChemPotn'])
    G_theory = pd.Series(G_theory, name='G_theory')
    delta_list = ['AllPotn', 'ChemPotn', 'PolChem', 'ApolChem', 'PartMolV', 'PolMolV', 'ApolMolV', 'Entropy', 'PolEnt', 'ApolEnt', 'SolPotn', 'PolSolPotn', 'ApolSolPotn', 'Solvent', 'PolSolvent', 'ApolSolvent', 'SASA']
    list_df_delta = []
    for data in delta_list:
        list_df_delta.append(delta(df, data))
    #new_df = pd.concat([df, sys_charge, sys_charge2, prt_charge2, lig_charge2] + list_df_delta + [G_theory, G_experiment], axis=1)
    new_df = pd.concat([df] + list_df_delta + [G_theory, G_experiment], axis=1)
    return new_df

def del_Enotrun(df, verbose):
    n1 = df.shape[0]
    df = df[df['Bond'].str.contains('not_run') == False]
    n2 = df.shape[0]
    if verbose:
        print(f'del not run MM: {n1 - n2}')
    for name in ['Sys', 'Prt', 'Lig']:
        n1 = n2
        df = df[df[f'{name}_Total'].str.contains('not_run') == False]
        n2 = df.shape[0]
        if verbose:
            print(f'del not run {name}: {n1 - n2}')
    return df

def read_efiles(data_files, verbose=0):
    dfs = []
    for fname in data_files:
        df = read_tsv(fname)
        df = del_Enotrun(df, verbose)
        df = df.astype(float)
        dfs.append(df)
    df = pd.concat(dfs)
    return df

def delta_df(df, name):
    df[f'delta_{name}'] = df[f'Sys_{name}'] - df[f'Prt_{name}'] - df[f'Lig_{name}']
    return df

def clean_epoten(df):
    #modify MM
    df['E'] = df['Bond'] + df['Angle'] + df['Dihed'] + df['Vdwaals'] + df['EEL'] + df['VDW-14'] + df['EEL-14']
    #rename columns
    df = df.rename(columns={'Vdwaals': 'VDW'})
    rename_cols = {}
    mm_col = ['E', 'Bond', 'Angle', 'Dihed', 'VDW', 'EEL', 'VDW-14', 'EEL-14', 'EGB', 'Restraints', 'EAmber']
    for col in mm_col:
        rename_cols[col] = f'MM_{col}'
    df = df.rename(columns=rename_cols)
    #modify Rism
    names = ['Sys', 'Prt', 'Lig']
    #create rism E
    for name in names:
        df[f'{name}_E'] = df[f'{name}_Total'] - df[f'{name}_3D-RISM']
    #create delta
    delta_col = ['E', 'Bond', 'Angle', 'Dihedral', 'LJ', 'Coulomb', 'LJ-14', 'Coulomb-14']
    for col in delta_col:
        df = delta_df(df, col)
    #modify MM Rism
    df['MM_E-Sys_E'] = (df['MM_E'] - df['Sys_E']).abs()
    return df
