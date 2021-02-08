import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image
import streamlit as st
st.write(""" DIABETES BASED ON MACHINE LEARNING AND PYTHON""")
image=Image.open('C:/Users/nlnna/OneDrive/Desktop/machinewebapp/Smile.jpg')
st.image(image, caption='Narayana', use_coloum_width=True)
df=pd.read_csv('C:/Users/nlnna/OneDrive/Desktop/machinewebapp/diabetes.csv')

# set a subheader
st.subheader('Data Information')
#show the data in table

st.dataframe(df)

#show the statistics on the data
st.write(df.describe())
# show the data as a chart

chart=st.bar_chart(df)

# split the data into independent  X and dependent y variables

x=df.iloc[:,0:8].values
y=df.iloc[:,-1].values

# split data set in to 75%  training and 25% testing

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.25, random_state=0)

#get the features input from the user

def get_user_input():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 0)
    Glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    BloodPressure = st.sidebar.slider('BloodPressure', 0, 122, 72)
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 99, 23)
    Insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.5)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.372)
    Age = st.sidebar.slider('Age', 0, 81, 29)

# store a dictionary into a veriable

    user_data = {
        'Pregnancies' : Pregnancies,
        'Glucose' : Glucose,
        'BloodPressure' : BloodPressure,
        'SkinThickness' : SkinThickness,
        'Insulin' : Insulin,
        'BMI' : BMI,
        'DiabetesPedigreeFunction' : DiabetesPedigreeFunction,
        'Age' : Age,
        }
# transform the data into a dataframe

    features = pd.DataFrame(user_data, index = [0])
    return features

#store the user input in to a variable

user_input = get_user_input()

# set a subheader and display the user input

st.subheader('user Input')

st.write(user_input)

# create and train the model

RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(x_train, y_train)

# show the model metrics

st.subheader('model test accuracy score:')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test))*100) + '%')

#store  the models predicts in a variables

prediction = RandomForestClassifier.predict(user_input)

#set a subheader and display the clasiffier

st.subheader('classification')
st.write(prediction)