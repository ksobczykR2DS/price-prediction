from skopt.space import Real, Categorical, Integer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


MODELS_DICT = {
    'DT': DecisionTreeRegressor(),
    'RF': RandomForestRegressor(),
    'SVR': SVR(),
    'LGB': LGBMRegressor(),
    'GBM': GradientBoostingRegressor(),
    'XGB': XGBRegressor()
}

MODELS_PARAMS = {
    'DT': [
        {
            'regressor_model__random_state': Categorical([0]),
            'regressor_model__max_depth': Integer(3, 17),
            'regressor_model__min_samples_leaf': Integer(10, 100),
            'regressor_model__min_samples_split': Integer(2, 50),
            'regressor_model__max_features': Categorical(['sqrt', 'log2', None]),
        }
    ],
    'RF': [
        {
            'regressor_model__random_state': Categorical([0]),
            'regressor_model__n_estimators': Integer(200, 1500),
            'regressor_model__max_depth': Integer(3, 17),
            'regressor_model__min_samples_leaf': Integer(10, 50),
            'regressor_model__min_samples_split': Integer(3, 20),
            'regressor_model__max_features': Categorical(['sqrt', 'log2', None])
        }
    ],
    'SVR': [
        {
            'regressor_model__random_state': Categorical([0]),
            'regressor_model__C': Real(10, 40, prior='log-uniform'),
            'regressor_model__gamma': Real(1e-4, 1e-1, prior='log-uniform'),
            'regressor_model__degree': Integer(1, 5),
            'regressor_model__kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
        }
    ],
    'GBM': [
        {
            'regressor_model__random_state': Categorical([0]),
            'regressor_model__n_estimators': Integer(2000, 6000),
            'regressor_model__learning_rate': Real(0.01, 0.05),
            'regressor_model__max_depth': Integer(2, 6),
            'regressor_model__max_features': Categorical(['sqrt']),
            'regressor_model__min_samples_leaf': Integer(10, 20),
            'regressor_model__min_samples_split': Integer(10, 20),
            'regressor_model__loss': Categorical(['huber'])
        }
    ],
    'LGB': [
        {
            'regressor_model__random_state': Categorical([0]),
            'regressor_model__num_leaves': Integer(5, 20),
            'regressor_model__n_estimators': Integer(2000, 6000),
            'regressor_model__learning_rate': Real(1e-2, 1e-1),
            'regressor_model__verbose': Integer(-10, 10),
            'regressor_model__max_bin': Integer(50, 250),
            'regressor_model__bagging_fraction': Real(0.1, 1),
            'regressor_model__bagging_freq': Integer(2, 10),
            'regressor_model__bagging_seed': Integer(1, 10),
            'regressor_model__feature_fraction': Real(0.1, 1),
            'regressor_model__feature_fraction_seed': Integer(1, 10),
            'regressor_model__objective': Categorical(['regression'])
        }],
    'XGB': [
        {
            'regressor_model__random_state': Categorical([0]),
            'regressor_model__n_estimators': Integer(2000, 6000),
            'regressor_model__learning_rate': Real(0.01, 0.05),
            'regressor_model__max_depth': Integer(2, 6),
            'regressor_model__min_child_weight': Real(0, 0.5),
            'regressor_model__subsample': Real(0.5, 1.0, prior='uniform'),
            'regressor_model__reg_alpha': Real(0.01, 1),
            'regressor_model__reg_lambda': Real(0, 1),
            'regressor_model__gamma': Real(0, 0.05),
            'regressor_model__colsample_bytree': Real(0.2, 1),
            'regressor_model__nthread': Categorical([-1])
        }
    ],
}
