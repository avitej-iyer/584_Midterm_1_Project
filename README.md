# 584_Midterm_1_Project

Group Members : Avitej Iyer, Siddharth Anmalsetty

The project explores three methods to predict which model generated a particular phrase - Logistic Regression, a pretrained model (we used BERT) and using LSTMs. Google drive links for the saved trained models for LSTM and BERT are given below. Ways to utilize these trained models are also presented below. The project was done in a colab file - the resulting ipynb is present in the repo. The code snippets below are also best run on Colab.

BERT trained model - [Drive link](https://drive.google.com/file/d/1aVrPPRMGvNT1ns89bP_ZAdCwEVwZo6cG/view?usp=sharing)

First, install dependencies and unzip the model - 
```
!pip install numpy transformers scikit-learn joblib tensorflow
!unzip "/content/bert_trained_model.zip" -d "/content/bert_text_classifier"
```

Then, you can use the model like so
```
import numpy as np
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder 
import joblib
import tensorflow as tf

# Load transformer model
loaded_bert_model = TFAutoModelForSequenceClassification.from_pretrained('/content/bert_text_classifier/content/bert_text_classifier')
loaded_tokenizer = AutoTokenizer.from_pretrained('/content/bert_text_classifier/content/bert_text_classifier')

# For new predictions - replace the text below with a new phrase
new_text = ["I spoke to my friend about a personal challenge I was facing, and their support and advice provided me with a new sense of clarity and direction."]

new_tokens = loaded_tokenizer(new_text, padding=True, truncation=True, return_tensors='tf', max_length=128)
predictions = loaded_bert_model.predict(dict(new_tokens))

# Get the index of the class with the highest logit (predicted class)
predicted_class_idx = np.argmax(predictions.logits, axis=1)

le = joblib.load("/content/bert_text_classifier/label_encoder.pkl")

# Decode the predicted class index to get the original label (model name)
predicted_model_name = le.inverse_transform(predicted_class_idx)

print(f"Predicted model: {predicted_model_name[0]}")
```

The trained LSTM model can be downloaded here - [Drive link](https://drive.google.com/file/d/1Ngz7ckWH7tKeHF16Q_ML0iFDrcUtId45/view?usp=sharing)

First, we install dependencies and extract our model : 
```
!pip install numpy transformers scikit-learn joblib tensorflow
!unzip "/content/lstm_trained_model.zip" -d "/content/lstm_trained_model"
```

And used as below, once it is uploaded to the session 
```
import numpy as np  # For numerical operations such as argmax
from tensorflow.keras.models import load_model  # For loading the LSTM model
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For padding sequences
import joblib  # For loading the saved LabelEncoder
import json 
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# Load the saved LSTM model
loaded_model = load_model('/content/lstm_trained_model/lstm_text_classifier_model.h5')

# Load the tokenizer from the JSON file
with open('/content/lstm_trained_model/tokenizer.json') as file:
    tokenizer_json = file.read()

# Reconstruct the tokenizer from the JSON
tokenizer = tokenizer_from_json(tokenizer_json)

# Inputing new text into the model - change the text below to try out new phrases
new_text = ["I spoke to my friend about a personal challenge I was facing, and their support and advice provided me with a new sense of clarity and direction."]
new_seq = tokenizer.texts_to_sequences(new_text)
new_padded = pad_sequences(new_seq, maxlen=30, padding='post')

# Use the loaded model to make predictions
predicted_class = loaded_model.predict(new_padded)

# Get the predicted class label (this will be in encoded form)
predicted_label = np.argmax(predicted_class, axis=1)

le = joblib.load("/content/lstm_trained_model/label_encoder.pkl")

# Convert the encoded label back to the original model name
predicted_model_name = le.inverse_transform(predicted_label)
print(f"Predicted model: {predicted_model_name[0]}")
```
