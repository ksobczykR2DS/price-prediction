import click
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBRegressor
import numpy as np

from data_utils import load_training_data

MODELS_DICT = {
    'LR': LinearRegression(),
    'DT': DecisionTreeRegressor(),
    'RF': RandomForestRegressor(),
    'SVR': SVR(),
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
            'regressor_model__C': Real(1e-6, 1e+6, prior='log-uniform'),
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
    'XGB': [
        {
            'regressor_model__random_state': Categorical([0]),
            'regressor_model__n_estimators': Integer(2000, 6000),
            'regressor_model__learning_rate': Real(0.01, 0.05),
            'regressor_model__max_depth': Integer(2, 6),
            'regressor_model__min_child_weight': Real(0, 0.5),
            'regressor_model__subsample': Real(0.5, 1.0, prior='uniform'),
            'regressor_model__reg_alpha': Real(0.01, 0.05),
            'regressor_model__gamma': Real(0, 0.05),
            'regressor_model__colsample_bytree': Real(0.03, 0.08),
            'regressor_model__nthread': Categorical([-1])
        }
    ],
}


@click.command()
@click.option('--model_names_list', required=True, type=str, help='list of models to run')
@click.option('--feature_engineering', required=True, is_flag=True, default=False,
              help='flag denoting use of engineered features')
@click.option('--n_iter', required=True, type=int, help='number of iterations for hyperparameters tuning')
def main(model_names_list, feature_engineering, n_iter):
    """
    Runs experiments with BayesSearch for selected models

    Usage:
    ```
    python run_experiments.py --model_names_list=DT,RF,SVR,GBM,XGB --feature_engineering --n_iter=10
    ```
    """
    print('Loading models...')
    model_dict = {}
    results = []
    try:
        for model_name in model_names_list.split(','):
            assert model_name in MODELS_DICT.keys(), f'Unknown model_name: {model_name}'
            model = MODELS_DICT[model_name]
            model_dict[model_name] = model
        x_train, x_test, y_train, y_test = load_training_data(feature_engineering=feature_engineering)
        for model_name, model in model_dict.items():
            print(f'Running experiments for {model_name}')
            pipe = Pipeline([('scaler', StandardScaler()), ('regressor_model', model)])
            opt = BayesSearchCV(
                pipe,
                MODELS_PARAMS[model_name],
                n_iter=n_iter,
                random_state=7,
                verbose=True
            )
            np.int = np.int64
            opt.fit(x_train, y_train)

            best_score = opt.score(x_test, y_test)
            best_params = opt.best_params_
            results.append({'Model': model_name, 'Score': best_score, **best_params})

    except Exception as e:
        print(f'An error occurred: {e}')

    results_df = pd.DataFrame(results)
    results_df.to_csv('reports/model_results.csv', index=False)


if __name__ == '__main__':
    main()
