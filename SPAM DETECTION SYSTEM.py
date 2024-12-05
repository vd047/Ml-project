 import tensorflow as tf
 from tensorflow.keras.preprocessing.sequence import pad_sequences
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.preprocessing.text import one_hot
 from tensorflow.keras.layers import LSTM
 from tensorflow.keras.layers import Dense
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Dropout, LSTM, Dense, Embedding
 from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
 import pandas as pd
 import matplotlib.pyplot as plt
 import nltk
 import re
 from nltk.corpus import stopwords
 from nltk.stem.porter import PorterStemmer
 from nltk.stem import WordNetLemmatizer
 from sklearn.metrics import confusion_matrix
 from sklearn.model_selection import train_test_split
 import numpy as np
 import seaborn as sns
 df=pd.read_csv("/content/Spam_SMS.csv")
 df=df.dropna()
 df
 label_counts = df['Class'].value_counts()
plt.figure(figsize=(6, 6))
 plt.pie(label_counts, labels=['ham', 'spam'], autopct='%1.1f%%', colors=['#66b3ff','#ff6666'
 plt.title('Percentage of Ham and Spam Reviews')
 plt.show()
 X=df.drop("Class",axis=1)
 y=df["Class"]
 message=X.copy()
 message
  nltk.download("stopwords")
   ps = PorterStemmer()
 message['Message'] = message['Message'].fillna('')
 corpus = []
 for i in range(len(message)):
    review = str(message.iloc[i, message.columns.get_loc("Message")])
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words("english")]
    review = " ".join(review)
    corpus.append(review)
    message["Message"][4]
 
 corpus[4]
  voc_size=1000
 one_hot_repair=[one_hot(word,voc_size) for word in corpus]
 one_hot_repair[:4]
 max_length = max(len(item) for item in one_hot_repair)
 print(max_length)
  max_length
 embedding_docs=pad_sequences(one_hot_repair,padding="pre",maxlen=max_length)
 embedding_docs[:4]
 from sklearn.preprocessing import LabelEncoder
 label_encoder = LabelEncoder()
 y = label_encoder.fit_transform(y) 
y[:4]
 embedding_vector_features = 30
 model = Sequential()
 model.add(Embedding(voc_size, embedding_vector_features))
 model.add(LSTM(64)) 
model.add(Dropout(0.3)) 
model.add(Dense(1, activation='sigmoid'))
 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
 reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
 X_final=np.array(embedding_docs)
 y_final=np.array(y)
 X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_
 history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
epochs=20, batch_size=32, 
callbacks=[early_stopping, reduce_lr])