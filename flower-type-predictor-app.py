# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

st.image('./iris1.png', width=800)

# Streamlit app
st.title("Iris Insight: Flower Type Predictor")

# Creator of this app
st.subheader("Created by: Aarish Asif Khan")
st.subheader("Created on: 10th January 2024")


st.markdown("The 'Iris Insight: Flower Type Predictor app' is a user-friendly tool designed to predict the type of Iris flower based on input parameters such as sepal length, sepal width, petal length, and petal width.")
st.markdown("Utilizing a Support Vector Machine (SVM) classifier trained on the famous Iris dataset, the app allows users to interactively input specific flower measurements through sliders in the sidebar")

st.subheader("How does the app work??")
st.markdown("Upon submission, the app processes the input, scales it for consistency, and provides a real-time prediction of the Iris flower type.") 
st.markdown("Additionally, the app displays the model's accuracy on a test set, offering users insights into the performance of the underlying machine learning model.")

# Sidebar for user input
st.sidebar.markdown("# User Input Parameters:")

def user_input_features():
    sepal_length = st.sidebar.slider(" #### Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    sepal_width = st.sidebar.slider("#### Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    petal_length = st.sidebar.slider(" #### Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
    petal_width = st.sidebar.slider(" #### Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
user_input = user_input_features()

# Show user input
st.subheader('User Input parameters:')
st.write(user_input)

# Preprocess user input
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
user_input_scaled = scaler.transform(user_input)

# Train the model with scaled data
model.fit(X_train_scaled, y_train)

# Make predictions
prediction = model.predict(user_input_scaled)

# Display the prediction
st.subheader('Making Predictions')
st.write(f'- ###### The anticipated type of Iris flower is: {iris.target_names[prediction[0]]}')

# Display model accuracy on the test set
y_pred = model.predict(scaler.transform(X_test))
accuracy = accuracy_score(y_test, y_pred)
st.subheader('Model Accuracy on Test Set')
st.write(f'- ###### The accuracy of the model is: {accuracy:.2f}')

# Display explanatory text
st.markdown("""
            ### What does this Web application do??
    * This app predicts the type of Iris flower based on user input parameters.
    * Use the sliders in the sidebar to input the sepal length, sepal width, petal length, and petal width.
    * The model is trained on the famous Iris dataset using a Support Vector Machine (SVM) classifier.
""")

# check the dataset
st.subheader("Iris Dataset:")

st.write('Here is the Iris dataset used in this web application. Feel free to download and explore it if you are interested.')
df = sns.load_dataset('iris')
st.write(df)

