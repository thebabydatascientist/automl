import pandas as pd
import datetime
import streamlit as st
import pycaret
from pycaret.classification import setup as sc,  compare_models as cmc,  evaluate_model as emc, plot_model as pmc,save_model as smc
from pycaret.regression import  *
import mlflow

def auto_train(dataset, ml_type, llm_label):

    data = dataset.copy()
    print(ml_type)
    if ml_type == 'classification':
        data.dropna(how='all', inplace=True)
        # Consistent tracking URI configuration:
        # mlflow.set_tracking_uri("mlflow/log")  # Set only once
        # mlflow.set_experiment("classify")
        # with mlflow.start_run():
        sc(data, target=llm_label, session_id=123,fix_imbalance = True, numeric_imputation='mean', categorical_imputation='mode') #,log_experiment = True, experiment_name = 'classify'
        best_model = cmc()
        st.markdown(f'Selected model: {best_model}')
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        best_model_path = f'model/best_model_{timestamp_str}'
        # st.write(plot_model(best_model, plot='residuals_interactive'))
        # st.write(plot_model(best_model, plot='feature'))
        smc(best_model, best_model_path)
        # mlflow.pycaret.log_model(best_model, "final_model")

    elif ml_type == 'regression':
        data.dropna(how='all', inplace=True)
        setup(data, target=llm_label, session_id=123,remove_outliers = True, numeric_imputation='mean', categorical_imputation='mode')
        best_model = compare_models()
        st.markdown(f'Selected model: {best_model}')
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        best_model_path = f'model/best_model_{timestamp_str}'
        # st.write(f'Model evaluation: {evaluate_model(best_model)}')
        save_model(best_model, best_model_path)

    else:
        st.warning("Please select other use case / label as we support only classification / regression at this point.")


    return best_model_path
