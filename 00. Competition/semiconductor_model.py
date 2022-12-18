# basic library
import gc
import argparse
import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# modeling library
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
import lightgbm
import xgboost


# hyperparameter-tuning library
import optuna
from optuna import Trial
from optuna.samplers import TPESampler


class EDA:
    def __init__(self) -> None:
        self.train_df = self.read_dataset('train.csv')
        self.test_df = self.read_dataset('test.csv')
        self.make_profile('train_pf.html', True)
        self.make_profile('test_pf.html', False)
        
    def read_dataset(self, name) -> pd.DataFrame:
        df = pd.read_csv(name)
        return df
    
    def make_profile(self, name, train=True) -> None:
        """
            pandas profiling 결과
            - 관측치 개수: 1,500개
            - 결측치 개수: 0개
            - 중복값 개수: 0개
            - 범주형 변수: 3개
            - 수치형 변수: 4개
        """
        if train:
            pf = self.train_df.profile_report()
            pf.to_file(name)
        else:
            pf = self.test_df.profile_report()
            pf.to_file(name)
        
    def check_duplicated_value(self) -> None:
        print(self.train_df.duplicated(['target']))
        print(self.train_df.duplicated(['target']).value_counts())
    
    def check_missing_value(self) -> None:
        """ 결측치 존재시 평균, 중앙, 최빈 값 등으로 대체 """
        print (f"[*] Train 결측치\n{self.train_df.isnull().sum()}")
        print (f"[*] Test 결측치\n{self.test_df.isnull().sum()}")
        
    def check_variable_dist(self) -> None:
        # 데이터 분포(평균, 중앙, 최빈값) 확인
        print("\n[X0]")
        print("[*] 평균: {}".format(self.train_df['X0'].mean()))
        print("[*] 중앙: {}".format(self.train_df['X0'].median()))
        print("[*] 최빈: {}".format(self.train_df['X0'].mode()))
        
        print("\n[X1]")
        print("[*] 평균: {}".format(self.train_df['X1'].mean()))
        print("[*] 중앙: {}".format(self.train_df['X1'].median()))
        print("[*] 최빈: {}".format(self.train_df['X1'].mode()))
        
        print("\n[X2]")
        print("[*] 평균: {}".format(self.train_df['X2'].mean()))
        print("[*] 중앙: {}".format(self.train_df['X2'].median()))
        print("[*] 최빈: {}".format(self.train_df['X2'].mode()))
        
        print("\n[X3]")
        print("[*] 평균: {}".format(self.train_df['X3'].mean()))
        print("[*] 중앙: {}".format(self.train_df['X3'].median()))
        print("[*] 최빈: {}".format(self.train_df['X3'].mode()))
        
        print("\n[X4]")
        print("[*] 평균: {}".format(self.train_df['X4'].mean()))
        print("[*] 중앙: {}".format(self.train_df['X4'].median()))
        print("[*] 최빈: {}".format(self.train_df['X4'].mode()))
        
        print("\n[X5]")
        print("[*] 평균: {}".format(self.train_df['X5'].mean()))
        print("[*] 중앙: {}".format(self.train_df['X5'].median()))
        print("[*] 최빈: {}".format(self.train_df['X5'].mode()))
        
        print("\n[Target]")
        print("[*] 평균: {}".format(self.train_df['target'].mean()))
        print("[*] 중앙: {}".format(self.train_df['target'].median()))
        print("[*] 최빈: {}".format(self.train_df['target'].mode()))
    
    def check_value_counts(self):
        print(self.train_df['X0'].value_coutns())
        print(self.train_df['X1'].value_coutns())
        print(self.train_df['X2'].value_coutns())
        print(self.train_df['X3'].value_coutns())
        print(self.train_df['X4'].value_coutns())
        print(self.train_df['X5'].value_coutns())
        print(self.train_df['target'].value_coutns())
        

class Feature:
    def __init__(self) -> None:
        self.train_df = self.read_csv('train.csv')
        self.test_df = self.read_csv('test.csv')
        
    def preprocess(self):
        # 수치형 데이터: 결측값 X & 이상값 X
        # 명목형 데이터: 레이블 인코딩, 원-핫 인코딩
        
        # 학습/테스트 데이터셋 분리
        X = self.train_df.iloc[:, :6]
        y = self.train_df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # 정규화
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
    

class Model:
    def __init__(self) -> None:
        self.xgb_param = {
            'eta': 0.001,
            'gamma' : 0,
            'max_depth' : 8,
            'n_estimators' : 50000,
            'sampling_method': 'gradient_based',
            'nthread': -1,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
        }
        
        self.lgbm_param = {
            'objective' : 'regression',
            'metric' : 'mae',
            'n_estimators' : 450000,
            'learning_rate' : 0.0001,
            'lambda_l1': 0.05,
            'lambda_l2': 0.05,
            'drop_rate': 0.5,
            'boost_from_average':False,
            'boosting': 'goss',
            'num_leaves': 62,
            'min_data_in_leaf': 40,
            # 'max_bin': 7,
            # 'bagging_faction':
            # 'bagging_req':
            # 'feature_faction':
            # 'max_depth':
            # 'path_smooth':
        }
        
        self.catboost_param = {
            'learning_rate':0.0001,
            'use_best_model':True,
            'early_stopping_rounds':1000, 
            'iterations':150000,
            'eval_metric':"MAE"
        }
        
        self.train_df = pd.read_csv('train.csv')
        self.test_df = pd.read_csv("test.csv")
    
        self.now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.xgb_model_name = f'xgboost_{self.now}.model'
        self.lgbm_model_name = f'lgbm_{self.now}.model'
        self.cb_model_name = f'catboost_{self.now}.model'
        self.skf = KFold(n_splits = 10, shuffle = True)
    
    def train_xgb(self):
        X = self.train_df.iloc[:, :6]
        y = self.train_df['target']
        model = XGBRegressor(**self.xgb_param)
        xgb_scaler = StandardScaler()
        xgb_mae = []
        for train_idx, valid_idx in self.skf.split(X=X, y=y):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10)
            X_train = xgb_scaler.fit_transform(X_train)
            X_valid = xgb_scaler.transform(X_valid)
            model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], early_stopping_rounds = 1000, verbose = 100, eval_metric = 'mae')
            y_pred = model.predict(X_valid)
            fold_mae = mean_absolute_error(y_valid, y_pred)
            xgb_mae.append(fold_mae)
            print("[XGB MAE] : {:.4f}\n".format(fold_mae))
        
        print(f"[XGB Average MAE] : {np.mean(xgb_mae)}")
        model.save_model(self.xgb_model_name)
        X_test = self.test_df.iloc[:, :6]
        X_test = xgb_scaler.transform(X_test)
        xgb_model = XGBRegressor(**self.xgb_param)
        xgb_model.load_model(self.xgb_model_name)
        y_pred = xgb_model.predict(X_test)
        self.test_df['target'] = y_pred
        self.test_df.to_csv(f'{self.xgb_model_name}.csv')
        
        
    def train_lgbm(self, X, y):
        X = self.train_df.iloc[:, :6]
        y = self.train_df['target']
        scaler = StandardScaler()
        model = LGBMRegressor(**self.lgbm_param)
        lgbm_mae = []
        for train_idx, valid_idx in self.skf.split(X=X, y=y):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.01)
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)
            model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], early_stopping_rounds = 1000, verbose = 100, eval_metric = 'mae')
            y_pred = model.predict(X_valid)
            fold_mae = mean_absolute_error(y_valid, y_pred)
            lgbm_mae.append(fold_mae)
            print("[LGBM MAE Error] : {:.4f}\n".format(fold_mae))
        
        print(f"[LGBM Average MAE = {np.mean(lgbm_mae)}")
        model.booster_.save_model('lightgbm.txt')
        X_test = self.test_df.iloc[:, :6]
        X_test = scaler.transform(X_test)
        lgbm_model = lightgbm.Booster(model_file='lightgbm.txt')
        y_pred = lgbm_model.predict(X_test)
        self.test_df['target'] = y_pred
        self.test_df.to_csv(f'{self.lgbm_model_name}.csv')
        
    
    def train_lgbm_all_dataset(self):
        X = self.train_df.iloc[:, :6]
        y = self.train_df['target']
        scaler = StandardScaler()
        model = LGBMRegressor(**self.lgbm_param)
        for train_idx in self.skf.split(X=X, y=y):
            X_train, y_train = X, y
            X_train = scaler.fit_transform(X_train)
            model.fit(X_train, y_train, verbose = 100, eval_metric = 'mae')
        
        model.booster_.save_model('lightgbm_all.txt')
        X_test = self.test_df.iloc[:, :6]
        X_test = scaler.transform(X_test)
        lgbm_model = lightgbm.Booster(model_file='lightgbm_all.txt')
        y_pred = lgbm_model.predict(X_test)
        self.test_df['target'] = y_pred
        self.test_df.to_csv(f'lightgbm_all.csv')
        
        
    def tune_hyperparameter(self, trial: Trial, X, y):
        param = {
            # 'n_estimators': trial.suggest_int('n_estimators', 150000, 450000),
            'n_estimators': trial.suggest_int('n_estimators', 300000, 500000),
            # 'max_depth': trial.suggest_int('max_depth', 8, 16),
            # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            'learning_rate': trial.suggest_float('learning_rate', low=0.00001, high=0.0005),
            # 'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
            'nthread': -1,
            # 'lambda_l1' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
            # 'lambda_l2': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            # 'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 1.0],)
        }
        
        model = LGBMRegressor(**param)
        lgbm_model = model.fit(X, y, verbose=True)
        score = mean_absolute_error(lgbm_model.predict(X), y)
        return score
        
    
    def train_catboost(self):
        X = self.train_df.iloc[:, :6]
        y = self.train_df['target']
        scaler = StandardScaler()
        model = CatBoostRegressor(**self.catboost_param)
        cb_mae = []
        for train_idx, valid_idx in self.skf.split(X=X, y=y):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10)
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)
            model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], early_stopping_rounds = 1000, verbose = 100)
            y_pred = model.predict(X_valid)
            fold_mae = mean_absolute_error(y_pred, y_valid)
            cb_mae.append(fold_mae)
            print("[CatBoost MAE Error] : {:.4f}\n".format(fold_mae))
        
        print(f"[CatBoost Average MAE] = {np.mean(cb_mae)}")
        model.save_model(self.cb_model_name)
        X_test = self.test_df.iloc[:, :6]
        X_test = scaler.transform(X_test)
        cb_model = CatBoostRegressor(learning_rate=0.0001, use_best_model=True, iterations=100000, eval_metric="MAE")
        cb_model.load_model(self.cb_model_name)
        y_pred = cb_model.predict(X_test)
        self.test_df['target'] = y_pred
        self.test_df.to_csv(f'{self.cb_model_name}.csv')
        
    
    def ensemble(self):
        X_test = self.test_df.iloc[:, :6]
        
        cb_model = CatBoostRegressor()
        cb_model.load_model("<model name>")
        cb_y_pred = cb_model.predict(X_test)
        
        lgbm_model = lightgbm.Booster(model_file='<model name>')
        lgbm_y_pred = lgbm_model.predict(X_test)
        
        xgb_model = XGBRegressor()
        xgb_model.load_model("<model name>")
        xgb_y_pred = xgb_model.predict(X_test)
        
        self.test_df['target'] = (lgbm_y_pred * 0.4) + (xgb_y_pred * 0.3) + (cb_y_pred * 0.3)
        self.test_df.to_csv(f'xgb_lgbm_cat_ensemble.csv')
        

if __name__ == "__main__":
    E = EDA()
    E.check_missing_value()
    E.check_duplicated_value()
    # F = Feature()
    # F.preprocess()
    M = Model()
    M.train_xgb()
    M.train_lgbm()
    M.train_catboost()
    M.ensemble()
    M.train_lgbm_all_dataset()
    
    train_df = pd.read_csv('train.csv')
    X = train_df.iloc[:, :6]
    y = train_df['target']
    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(lambda trial : M.tune_hyperparameter(trial, X, y), n_trials = 50)
    print('Best trial : score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))