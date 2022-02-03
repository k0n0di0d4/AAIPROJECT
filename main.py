import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, plot_confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import BayesianRidge
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from yellowbrick.model_selection import LearningCurve

#Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression
def Randomized_Forest():
  # get a dataset from file
  dataset = pd.read_csv('heart.csv')

  # view first 5 lines
  dataset.head(5)

  # dataset.info()
  X = dataset.drop(columns='output')
  Y = dataset['output']

  #intialize empty array and set best_precision to 0
  precision_array = []
  best_precision = 0

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
  model1 = RandomForestClassifier()
  hist = model1.fit(X_train, Y_train)

  for i in range(len(dataset.index)):
    Y_pred = model1.predict(X_test)
    precision = accuracy_score(Y_test, Y_pred)
    precision_array.append(precision)
    if precision > best_precision:
      best_precision = precision

  #print out the the most accurate precision
  print(best_precision)

  #print the whole trained array
  print(precision_array)

  # Learning curve plot
  visualizer = LearningCurve(model1, scoring='r2')
  visualizer.fit(X, Y)
  visualizer.show()

  #Confusion Matrix
  plot_confusion_matrix(model1, X_test, Y_test)
  plt.show()

  #ROC and AUC curve
  y_probabilities = model1.predict_proba(X_test)[:, 1]
  false_positive_rate, true_positive_rate, threshold = roc_curve(Y_test, y_probabilities)
  plt.figure(figsize=(10, 6))
  plt.title('ROC for decision tree')
  plt.plot(false_positive_rate, true_positive_rate, linewidth=5, color='green')
  plt.plot([0, 1], ls='--', linewidth=5)
  plt.plot([0, 0], [1, 0], c='.5')
  plt.plot([1, 1], c='.5')
  plt.text(0.2, 0.6, 'AUC: {:.2f}'.format(roc_auc_score(Y_test, y_probabilities)), size=16)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.show()

  return hist



def Bayesian_Regression():
  #load the dataset into variables
  datasett = pd.read_csv('heart.csv')
  X = datasett.drop(columns='output')
  Y = datasett['output']

  precision_array = []
  best_precision = 0
  # Train and test variables
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1)

  # Creating and training model
  model = BayesianRidge()
  hist = model.fit(X_train, Y_train)

  for i in range(len(datasett.index)):
    # Predicting based on test data
    Y_prediction = model.predict(X_test)
    precision = mean_absolute_error(Y_test, Y_prediction)
    precision = 1 - precision
    precision_array.append(precision)
    if precision > best_precision:
      best_precision = precision

  print(best_precision)
  print(precision_array)

  #Learning curve plot
  visualizer = LearningCurve(model, scoring='r2')
  visualizer.fit(X, Y)
  visualizer.show()
  return hist

def Neural_Network():
  dataset = pd.read_csv('heart.csv')

  # dataset.info()
  X = dataset.drop(columns='output')
  Y = dataset['output']

  min_max_scaler = preprocessing.MinMaxScaler()
  X_scale = min_max_scaler.fit_transform(X)

  X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
  X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

  model = Sequential([
    Dense(32, activation='relu', input_shape=(13,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
  ])
  model.compile(optimizer='sgd',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  #training the data
  hist = model.fit(X_train, Y_train,
                   batch_size=32, epochs=100,
                   validation_data=(X_val, Y_val))
  accuracy = model.evaluate(X_test, Y_test)[1]
  print(accuracy)

  #model accuracy and prediction plot
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper right')
  plt.show()
  return hist


forest_history = Randomized_Forest()
bayesian_history = Bayesian_Regression()
neu_net_history = Neural_Network()

plt.title("Accuracy")

plt.plot(neu_net_history.history['accuracy'], label = 'Neural Network')
plt.ylim(0, 1)
plt.show()