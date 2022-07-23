import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr

class AllData:
    def __init__(self, all_data, t_data, a_data=['G_experiment']): 
        #separate features, answer
        self.all_data_t = all_data[t_data].as_matrix()
        self.all_data_a = all_data[a_data].as_matrix()
        self.all_data_index = all_data.index.values
        #separate train, test
        train_t, test_t, train_a, test_a , train_index, test_index = train_test_split(data_t, data_a, index_list, test_size=0.2, random_state=2528)
        self.train_t = train_t
        self.test_t = test_t
        self.train_a = train_a
        self.test_a = test_a
        self.train_index = list(train_index)
        self.test_index = list(test_index)
        #others
        self.num_split = 5
        self.kf = KFold(n_splits=self.num_split, shuffle=True, random_state=2525)
        self.scale_data()


    def scale_data(self):
        scaler = StandardScaler()
        scaler.fit(self.all_data_t)
        self.train_t = scaler.transform(self.train_t)
        self.test_t = scaler.transform(self.test_t)

    def GSCV(self, model, grid_params, metric, refit=None, GSCPU=1, verbose=0):
        if refit is None:
            refit = metric[0]
        gscv = GridSearchCV(model, grid_params, cv=self.kf, scoring=metric,
            n_jobs=GSCPU, iid=False, refit=refit, verbose=verbose)
        gscv.fit(self.train_t, self.train_a)
        return gscv


    def my_cv_score(self, estimator, x, y):
        y_predicted = estimator.predict(x)
        y_predicted = list(y_predicted)
        y = y.T[0]
        r, p_value = pearsonr(y, y_predicted)
        return r

    def conf_shape(self, array):
        if len(array.shape) == 1:
            return array
        else:
            return array.T[0]


    def get_test_score(self, model):
        result_train = self.conf_shape(model.predict(self.train_t))
        list_train_a = self.train_a.T[0]
        predicted_result = self.conf_shape(model.predict(self.test_t))
        list_test_a = self.test_a.T[0]
        train_r, train_r_p = pearsonr(list_train_a, result_train)
        train_rho, train_rho_p = spearmanr(list_train_a, result_train)
        train_rmse = np.sqrt(mean_squared_error(list_train_a, result_train))
        test_r, test_r_p = pearsonr(list_test_a, predicted_result)
        test_rho, test_rho_p = spearmanr(list_test_a, predicted_result)
        test_rmse = np.sqrt(mean_squared_error(list_test_a, predicted_result))
        self.predicted_result = predicted_result
        self.list_test_a = list_test_a
        return (train_r, train_rho, train_rmse), (test_r, test_rho, test_rmse)


    def get_index(self):
        return self.train_index, self.test_index


    def train_xgb(self, model):
        model.fit(self.train_t, self.train_a, eval_set=[(self.train_t, self.train_a, 'train'), \
            (self.test_t, self.test_a, 'test')], eval_metric=self.my_score, verbose=False)


    def train_model(self, model):
        model.fit(self.train_t, self.train_a, eval_set=[(self.train_t, self.train_a, 'train'), \
            (self.test_t, self.test_a, 'test')], eval_metric=self.my_score, verbose=False)

    def save_model(self, model, save_model_path):
        model.save_model(save_model_path)

    def load_model(self, model, load_model_path):
        model.load_model(load_model_path)

    def update_model(self, model, save_model_path):
        updated_model = model.fit(self.train_t, self.train_a, xgb_model=save_model_path, \
            eval_set=[(self.train_t, self.train_a, 'train'), (self.test_t, self.test_a, 'test')], \
            eval_metric=self.my_score, verbose=False)

    def get_gap_values(self):
        gap_list = list(map(lambda x, y: abs(x-y), self.list_test_a, self.predicted_result))
        df = pd.DataFrame({'test_a':self.list_test_a, 'test_predict':self.predicted_result, \
            'gap_values':gap_list}, index=self.test_index)
        test_df = self.test_data
        df = pd.concat([test_df, df], axis=1)
        df = df.sort_values('gap_values')
        return df


    def my_score(self, y_predicted, y_true):
        y_predicted = list(y_predicted)
        y_true = y_true.get_label()
        r, p_value = pearsonr(y_true, y_predicted)
        return 'my-score', r

    def ave_cv_evals(self, model):
        evals_result_li = []
        for i_train, i_test in self.kf.split(self.train_t):
            x_cv_train = self.train_t[i_train]
            y_cv_train = self.train_a[i_train]
            x_cv_test = self.train_t[i_test]
            y_cv_test = self.train_a[i_test]
            eval_set = [(x_cv_test, y_cv_test)]
            model.fit(x_cv_train, y_cv_train, eval_set=eval_set, eval_metric=self.my_score, verbose=False)
            evals_result = model.evals_result()
            evals_result_li.append(evals_result)
        num_boost = len(evals_result_li[0]['validation_0']['my-score'])
        ave_cv_evals = {'my-score': np.zeros(num_boost),
                             'rmse': np.zeros(num_boost)}
        for i in range(self.num_split):
            ave_cv_evals['my-score'] += evals_result_li[i]['validation_0']['my-score']
            ave_cv_evals['rmse'] += evals_result_li[i]['validation_0']['rmse']
        ave_cv_evals['my-score'] /= self.num_split
        ave_cv_evals['rmse'] /= self.num_split
        return ave_cv_evals


class TrainTestData(AllData):
    def __init__(self, trainfname, testfname, t_data, a_data=['G_experiment'], work_dir='./'):
        self.train_data = pd.read_csv(trainfname, index_col=0)
        self.test_data = pd.read_csv(testfname, index_col=0)
        self.all_data = pd.concat([self.train_data, self.test_data], axis=0)

        print(f'all data: {self.all_data.shape}')
        print(f'train data: {self.train_data.shape}')
        print(f'test data: {self.test_data.shape}')

        ##shuffle
        self.train_data = self.train_data.sample(frac=1, random_state=2525)
        self.test_data = self.test_data.sample(frac=1, random_state=2525)
        #separate feature, answer
        self.all_data_t = self.all_data[t_data].as_matrix()
        self.all_data_a = self.all_data[a_data].as_matrix()
        ##train
        self.train_t = self.train_data[t_data].as_matrix()
        self.train_a = self.train_data[a_data].as_matrix()
        self.train_index = list(self.train_data.index.values)
        ##test
        self.test_t = self.test_data[t_data].as_matrix()
        self.test_a = self.test_data[a_data].as_matrix()
        self.test_index = list(self.test_data.index.values)
        #others
        self.num_split = 5
        self.kf = KFold(n_splits=self.num_split, shuffle=True, random_state=2525)
        self.scale_data()
        

    def split_train_test(self, fi_name):
        with open(fi_name, 'r') as f:
            datas = f.read().strip('\n').replace(' ', '')
        pdbid_li = datas.split(',')
        print(f'The number of test set pdbid: {len(pdbid_li)}')
        no_pdbid_li = []
        train_data = self.all_data.copy()
        test_data = pd.DataFrame(columns=train_data.columns)
        for pdbid in pdbid_li:
            new_data = train_data[train_data['PdbId'] == pdbid]
            if new_data.empty:
                no_pdbid_li.append(pdbid)
                continue
            test_data =test_data.append(new_data)
            train_data = train_data[train_data['PdbId'] != pdbid]
        print(f'all data: {self.all_data.shape}')
        print(f'train data: {train_data.shape}')
        print(f'test data: {test_data.shape}')
        return train_data, test_data, no_pdbid_li

