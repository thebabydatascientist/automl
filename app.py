from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
from train import auto_train
from llm_module import assign_prediction_type, assign_label
from PIL import Image
import time
import ydata_profiling as pp
from streamlit_pandas_profiling import st_profile_report
import openai
# import mlflow

# Set the page layout to "wide"
st.set_page_config(layout="wide")

if "model_path" not in st.session_state:
    st.session_state.model_path = ""
if "dataset" not in st.session_state:
    st.session_state.dataset = ""
if "describe_problem" not in st.session_state:
    st.session_state.describe_problem = ""
if "path_to_data" not in st.session_state:
    st.session_state.path_to_data = None
if "model_path" not in st.session_state:
    st.session_state.model_path = None
if "ml_type" not in st.session_state:
    st.session_state.ml_type = ""
if "llm_label" not in st.session_state:
    st.session_state.llm_label = ""
if "eda" not in st.session_state:
    st.session_state.eda = ""
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "data" not in st.session_state:
    st.session_state.data = None
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

col1, col2 = st.columns(2)
with col1:
    st.title(f"Auto AI Tool")
with col2:
    image = Image.open('images/logo.png')
    st.image(image, use_column_width=False, width  = 150)


dynamic_image = Image.open('images/dynamic_image.jpg')
st.sidebar.image(dynamic_image)

def progress_bar():
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

def preprocess(describe_problem,dataset ):
    columns = dataset.columns
    ml_type = assign_prediction_type(describe_problem,columns )
    st.session_state.ml_type = ''.join(ml_type)

    llm_label = assign_label( st.session_state.ml_type , columns)
    llm_label = ''.join(llm_label)


    return st.session_state.ml_type  ,llm_label

def train(dataset, ml_type,llm_label ):
    progress_bar()
    st.session_state.model_path = auto_train(dataset, ml_type,llm_label)
    st.success("Model Saved")
    return st.session_state.model_path


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run(model_path, data ):
    model = load_model(model_path)
    predictions = predict_model(estimator=model, data=data)
    return predictions

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



st.session_state.path_to_data = st.file_uploader("Upload csv file for training the model", type=["csv"])
if st.session_state.path_to_data is not None:
    st.session_state.dataset = pd.read_csv(st.session_state.path_to_data)
    st.subheader('Preview the first lines of the dataset:')
    st.write(st.session_state.dataset.head())

st.session_state.describe_problem = st.text_input( "Please describe in a few words the problem you're trying to solve and the dataset.")

if st.session_state.ml_type != '':
    st.sidebar.info(f'This app will create a  {st.session_state.ml_type} model.')

if st.sidebar.button("Explore dataset - optional",type="secondary"):

    with st.expander(label = 'EDA', expanded=False):
        st.session_state.eda = st.session_state.dataset.profile_report(title="Profiling Report")
        st_profile_report(st.session_state.eda)

if st.sidebar.button("1. Prepare dataset", type="primary"):
    if st.session_state.path_to_data is not None:
        st.session_state.ml_type ,st.session_state.llm_label = preprocess(st.session_state.describe_problem,  st.session_state.dataset)
        progress_bar()

if st.session_state.ml_type != '' and st.session_state.llm_label != '':
    st.write(f' AI selected  {st.session_state.llm_label} as the target variable to be predicted. Please select it if correct, or choose your own. ')
    st.session_state.llm_label = st.selectbox('Please select the label to be predicted:',st.session_state.dataset.columns , index = st.session_state.dataset.columns.get_loc(st.session_state.llm_label))

if st.sidebar.button("2. Train Model",type="primary"):
    if st.session_state.ml_type != '' and st.session_state.llm_label != '':
        progress_bar()
        train(st.session_state.dataset, st.session_state.ml_type,   st.session_state.llm_label )
    else:
        st.error('Train the Model first!')

file_upload2 = st.file_uploader("Upload csv file for predictions", type=["csv"])
if file_upload2 is not None:
    st.session_state.data = pd.read_csv(file_upload2)

if st.sidebar.button("3. Predict on new data", type="primary"):
    if st.session_state.model_path is not None and file_upload2 is not None :
        progress_bar()
        st.session_state.predictions = run(st.session_state.model_path,st.session_state.data)
        st.session_state.predictions["feedback"] = "correct"
    else:
        st.error('Train the Model first!')

if st.session_state.predictions is not None:
    predictions_feedback = st.data_editor(st.session_state.predictions , num_rows='dynamic')
    st.download_button(label="download predictions", data=predictions_feedback.to_csv(), file_name="predicted.csv",
                   mime="text/csv", type="primary")


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Simulate stream of response with milliseconds delay
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            #will provide lively writing
            stream=True,
        ):
            #get content in response
            full_response += response.choices[0].delta.get("content", "")
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
