import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def assign_prediction_type(user_input, features):
    messages_template = [
        {
            "role": "system",
            "content": """You are an AI expert system specialised in determining the type of machine learning algorithm to be applied based on a give problem statement.
                       Workflow: User provides you with a problem statement and features of dataset.
                       2.You return the applicable algorithm type from list: classification or regression. \nRULES:\n Return only one word, nothing else."""
        },
        {
            "role": "user",
            "content": f"Problem statement: [{user_input, features}]."
        },
        {
            "role": "assistant",
            "content": "classification"
        }
    ]
    new_message = {
        "role": "user",
        "content": f"Problem statement: [{user_input}],Features: [{features}]."
    }
    messages_template.append(new_message)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=messages_template,
        temperature=0.1,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    return response["choices"][0]['message']['content'].split("\n")

def assign_label( ml_type, datainfo):
    messages_template = [
        {
            "role": "system",
            "content": f"""As an AI expert system, I specialize in selecting the optimal 
            feature for prediction based on a given machine learning task and dataset columns. 
            I supply you with the type of machine learning task and the 
            dataset columns, and you'll provide you with the most suitable column to serve as the label for prediction. \nRULES:\n Return only one word, it should be a column from provided data, nothing else."""
        },
        {
            "role": "user",
            "content": f"ML type and dataset information: [{ ml_type,datainfo}]."
        }
    ]
    new_message = {
        "role": "user",
        "content": f"ML type , data columns: [{ml_type,datainfo}]."
    }
    messages_template.append(new_message)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=messages_template,
        temperature=0.1,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    return response["choices"][0]['message']['content'].split("\n")

# response =assign_prediction_type("""The dataset contains these columns N = Nitrogen
# P = phosphorous
# K = Potassium
# Temperature=The average soil temperatures for bioactivity range from 50 to 75F.
# Ph = A scale used to identify acidity or basicity nature; (Acid Nature- Ph<7; Neutral- Ph=7; Base Nature-P>7)
# label = Types of Crop (Rice,Maize, Chickpea; Kidney beans; pigeonpeas; mothbeans; mungbean;blackgram; lentil; pomegranate; banana; mango; grapes; watermelon; muskmelon; apple; orange;papaya; coconut; cotton; jute; coffee). Help me predict the label.""")
#
# print(response)