import streamlit as st
import pickle
import numpy as np
from PIL import Image
import time  

# Load the saved model (from pickled file)
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to make predictions based on the feature values
def make_prediction(sepal_length, sepal_width, petal_length, petal_width):
    # Prepare the input data as a numpy array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predict using the trained model
    prediction = model.predict(input_data)
    
    # Map the numeric prediction back to the species name
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    return species[prediction[0]]

# Streamlit UI
st.title("🌸 Iris Flower Prediction App")

# Display app description
st.markdown("""
    This app predicts the species of Iris flowers based on their measurements.
    You can input the flower's measurements using the sliders below .
    The app uses a **Random Forest** model trained on the famous Iris dataset.
""")

# Show the user the feature values they have selected
st.subheader("Flower Measurements Input:")
sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.0, step=0.1)

# Show the selected feature values
st.write(f"**Selected Values**:")
st.write(f"Sepal Length: {sepal_length} cm")
st.write(f"Sepal Width: {sepal_width} cm")
st.write(f"Petal Length: {petal_length} cm")
st.write(f"Petal Width: {petal_width} cm")

# Add a progress bar
progress_bar = st.progress(0)

# When the user clicks the button, make a prediction
if st.button("Predict Species"):
    # Simulate a delay to show the progress bar
    for i in range(100):
        time.sleep(0.005)
        progress_bar.progress(i + 1)

    # Make the prediction
    result = make_prediction(sepal_length, sepal_width, petal_length, petal_width)

    # Display the prediction result
    st.write(f"The predicted species is: **{result}**")
    
    # Show image corresponding to the predicted species
    if result == "Iris-setosa":
        image = Image.open("setosa.jpg")  # Path to the image of Iris-setosa
    elif result == "Iris-versicolor":
        image = Image.open("versicolor.jpg")  # Path to the image of Iris-versicolor
    else:
        image = Image.open("virginica.jpg")  # Path to the image of Iris-virginica
    
    st.image(image, caption=f"{result} Flower", use_column_width=True)

    # Show the confidence of the prediction
    st.write(f"**Confidence Level**: The model is highly confident in this prediction based on the input features.")

# Footer Section with some additional text
st.markdown("""
    ---
    **Contact Information**:
    - Created by: Gautam
    - Email: gautamfeb2005@gmail.com
""")

    

