from data_utils import load_training_data
from models_params_dict import MODELS_DICT, MODELS_PARAMS
import click
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    python run_experiments.py --model_names_list=DT,RF,SVR,LGB,GBM,XGB --feature_engineering --n_iter=50
    ```
    """
    print('Loading models...')
    model_dict = {}

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

            result_df = pd.DataFrame([{**{'Model': model_name, 'Score': best_score}, **best_params}])
            result_df.to_csv(f'reports/model_results_{model_name}.csv', index=False)

    except Exception as e:
        print(f'An error occurred: {e}')
