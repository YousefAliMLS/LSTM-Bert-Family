import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter

data = pd.read_csv('cellula toxic data  (1).csv')
X = data['query'].values
Target = data['Toxic Category'].values

labelEncoder = LabelEncoder()
label_Encoded = labelEncoder.fit_transform(label)
uni_classed = len(labelEncoder.classes_)
y = to_categorical(label_Encoded)

tokenizer = Tokenizer(num_words = 1000)
tokenizer.fit_on_texts(text)
seq = tokenizer.texts_to_sequences(text)
X = pad_sequences(seq, maxlen = 100)

X_train , X_test, y_train, y_test =  train_test_split(X , y , test_size=0.3, random_state=40)

model = Sequential()
model.add(Embedding(input_dim= 10000, output_dim = 128, input_length = 100))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(uni_classed, activation='softmax'))
model.compile(optimizer = 'adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

HISTORY = model.fit (X_train, y_train, epochs = 10, batch_size = 64, validation_split = 0.2)

y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_true_class = np.argmax(y_test, axis= 1)

f1 = f1_score(y_true_class, y_pred_class, average='weighted')
report = classification_report(y_true_class, y_pred_class, target_names=labelEncoder.classes_)

cm = confusion_matrix(y_true_class, y_pred_class)

pdf_path = r"C:\Users\Al Badr\Desktop\Cellula_1week_[Yousef_Mahmoud_Ali]\LSTM\LSTM_results.pdf"

with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labelEncoder.classes_, yticklabels=labelEncoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    pdf.savefig()
    plt.close()

    plt.plot(HISTORY.history['accuracy'], label='Training Accuracy')
    plt.plot(HISTORY.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.plot(HISTORY.history['loss'], label='Training Loss')
    plt.plot(HISTORY.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    pdf.savefig()
    plt.close()


buffer = BytesIO()
c = canvas.Canvas(buffer, pagesize=letter)
textobject = c.beginText(50, 750)
textobject.textLine("Classification Report:")
for line in report.split("\n"):
    textobject.textLine(line)
textobject.textLine("")
textobject.textLine(f"Weighted F1 Score: {f1:.4f}")
c.drawText(textobject)
c.save()

buffer.seek(0)
writer = PdfWriter()


for page in PdfReader(pdf_path).pages:
    writer.add_page(page)


for page in PdfReader(buffer).pages:
    writer.add_page(page)


with open(pdf_path, "wb") as f_out:
    writer.write(f_out)
