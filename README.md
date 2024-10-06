# 584_Midterm_1_Project

The project explores three methods to predict which model generated a particular phrase - Logistic Regression, a pretrained model (we used BERT) and using LSTMs. Google drive links for the saved trained models for LSTM and BERT are given below. Ways to utilize these trained models are also presented below.

BERT trained model - [Drive link](https://drive.google.com/file/d/1aVrPPRMGvNT1ns89bP_ZAdCwEVwZo6cG/view?usp=sharing)

First, install dependencies and unzip the model - 
```
!pip install numpy transformers scikit-learn joblib tensorflow
!unzip "/content/bert_trained_model.zip" -d "/content/bert_text_classifier"
```

Then, you can use the model like so
```
import numpy as np  # For numerical operations such as argmax
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer  # For loading the BERT model and tokenizer
from sklearn.preprocessing import LabelEncoder  # Assuming 'le' is a label encoder for decoding predictions
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

The trained LSTM model can be downloaded here - [Drive link](https://drive.google.com/file/d/19QJj6uX3yxuINm-Bu8Sjz_22t_OrRyeJ/view?usp=sharing)

And used as below, once it is uploaded to the session 
```
#loading and using the lstm model
from tensorflow.keras.models import load_model

# Load the saved LSTM model
loaded_model = load_model('lstm_text_classifier_model.h5')

# Inputing new text into the model - change the text below to try out new phrases
new_text = ["I spoke to my friend about a personal challenge I was facing, and their support and advice provided me with a new sense of clarity and direction."]
new_seq = tokenizer.texts_to_sequences(new_text)
new_padded = pad_sequences(new_seq, maxlen=30, padding='post')

# Use the loaded model to make predictions
predicted_class = loaded_model.predict(new_padded)

# Get the predicted class label (this will be in encoded form)
predicted_label = np.argmax(predicted_class, axis=1)

# Convert the encoded label back to the original model name
predicted_model_name = le.inverse_transform(predicted_label)
print(f"Predicted model: {predicted_model_name[0]}")
```
