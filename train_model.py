import numpy as np
import pandas as pd
import tensorflow as tf


from model_results import get_results
from preprocess import get_train_test_split, process_data
from models.create_model import NN_BASE, NN_LSTM, NN_LSTM_DROPOUT

df = pd.read_csv("data/fake_job_postings_v4.csv", encoding='cp1252')
data, labels = process_data(df)

num_folds = 1
split = pd.read_csv('data/10-cv.index.csv')['x']
acc = np.zeros((num_folds))
sen = np.zeros((num_folds))
spec = np.zeros((num_folds))
prec = np.zeros((num_folds))
f1 = np.zeros((num_folds))

for n in range(num_folds):
    train, test = get_train_test_split(np.array(split), n + 1)
    X_train, y_train = data[train], np.array(labels[train])
    X_test, y_test = data[test], np.array(labels[test])
    
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc'),
    ]

    model = NN_BASE(data, metrics)
    # model = NN_LSTM(data, metrics)
    # model = NN_LSTM_DROPOUT(data, metrics)
    print(model.summary())

    epochs = 10
    batch_size = 128
    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, validation_split=0.1, verbose=2)

    accuracy, sensitivity, specificity, precision, f1_score = get_results(model, X_test, y_test)
    acc[n] = (accuracy)
    sen[n] = (sensitivity)
    spec[n] = (specificity)
    prec[n] = (precision)
    f1[n] = (f1_score)

print('Accuracy', np.mean(acc), np.std(acc))
print('Sensitivity', np.mean(sen), np.std(sen))
print('Specificity', np.mean(spec), np.std(spec))
print('Precision', np.mean(prec), np.std(prec))
print('F1-Score', np.mean(f1), np.std(f1))
