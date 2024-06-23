## Pokemon Face Recognition App
This project aims to create a Pokemon Face Recognition application using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The model is trained to classify different Pokemon faces and can predict the Pokemon from a given image.

## Table of Contents
- [Installation](#installation)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Predicting the Pokemon](#predicting-the-pokemon)
- [Usage](#usage)

## Installation

Clone the repository:
```bash
git clone https://github.com/SUDAR2005/pokemon-face-app.git
cd pokemon-face-app
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8
- TensorFlow 2.x
- Keras
- NumPy

You can install all the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset used for training the model contains images of various Pokemon faces categorized into different folders. The dataset should be structured as follows:
```
pokemon_dataset/
    Bulbasaur/
        bulbasaur1.jpg
        bulbasaur2.jpg
        ...
    Pikachu/
        pikachu1.jpg
        pikachu2.jpg
        ...
    ...
```

You can download the dataset from the [GitHub repository](https://github.com/SUDAR2005/pokemon-face-app/tree/main/pokemon_dataset).

## Training the Model

The model can be trained using the `train_model` function defined in the `pokemon_face_app.ipynb` file. This function uses TensorFlow and Keras to build and train a CNN model on the provided dataset.

To train the model, run the following code:
```python
from pokemon_face_app import train_model

train_model('./pokemon_dataset')
```

This will train the model and save it to `./model/pokemon_model.h5`.

## Predicting the Pokemon

To predict the Pokemon from a given image, use the `predict_pokemon` function. This function loads the trained model and predicts the class of the input image.

Example usage:
```python
from pokemon_face_app import predict_pokemon

user_photo_path = './path/to/prediction/img.jpg'
predicted_pokemon, text_output = predict_pokemon(user_photo_path)

print("Text Output:", text_output)
print("Predicted Pokemon:", predicted_pokemon)
```

## Usage

1. **Mount Google Drive (if using Google Colab):**
    ```python
    from google.colab import drive
    drive.mount('/content/drive', True)
    ```

2. **Train the model:**
    ```python
    from pokemon_face_app import train_model
    train_model('./pokemon_dataset')
    ```

3. **Predict the Pokemon:**
    ```python
    from pokemon_face_app import predict_pokemon

    user_photo_path = './path/to/prediction/img.jpg'
    predicted_pokemon, text_output = predict_pokemon(user_photo_path)

    print("Text Output:", text_output)
    print("Predicted Pokemon:", predicted_pokemon)
    ```

## File Structure

- `pokemon_face_app.ipynb`: Jupyter Notebook containing the code for training and predicting Pokemon faces.
- `requirements.txt`: List of required packages.
- `pokemon_dataset/`: Directory containing the dataset of Pokemon images.
