from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd

model = load_model('deployment_28062021')

def predict(model, input_df):
    predictions_data = predict_model(estimator=model, data=input_df)
    predictions = predictions_data['Label'][0]
    return predictions

st.title('Diamond Price Prediction Web App')
st.write('This is a web app to predict the price of the Diamond based on the caret,'
         'quality and size of the Diamond, and several features that you can see in the sidebar. '
         'Please adjust the value of each feature. After that, click on the Predict button to see the prediction '
         'of the Regression')
st.write('THIS MODEL HAS AN ACCURACY VALUE OF 98.01%')

carat = st.sidebar.slider(label='Carat', min_value=0.2, max_value=5.01, value=3.00, step=0.01)

cut = st.sidebar.selectbox("Cut Quality",['Fair','Good','Very Good','Premium','Ideal'])

color = st.sidebar.selectbox("Color(Best to Worst)", ['D','E','F','G','H','I','J'])

clarity = st.sidebar.selectbox("Clarity(Best to Worst)",['I1','SI2','VS2','VS1','VVS2','VVS1','IF'])

table = st.sidebar.slider("Table(Top of Diamond in mm)", min_value=43.00, max_value=95.00, value=65.00, step=0.01)

x = st.sidebar.slider(label='Length(mm)', min_value=0.00, max_value=10.74, value=5.00, step=0.01)

y = st.sidebar.slider(label='Width(mm)', min_value=0.00, max_value=58.90, value=30.00, step=0.01)

z = st.sidebar.slider(label='Depth(mm)', min_value=0.00, max_value=31.80, value=15.00, step=0.01)

output = ""

features = {'carat': carat, 'cut': cut, 'color': color,
            'table': table,'clarity': clarity, 'x': x, 'y': y, 'z': z
            }

features_df = pd.DataFrame([features])

st.table(features_df)

if st.button('PREDICT'):

    output = predict_model(model, features_df)
    output = '₹' + str(output)

    st.success('The predicted price is {}'.format(output))
