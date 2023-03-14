import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X_train = np.array([[1400], [1600], [1700], [1875], [1100], [1550], [2350], [2450], [1425], [1700]])
y_train = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

# Creating linear regression model
model = LinearRegression()

# Training the model using the training data
model.fit(X_train, y_train)

# Testing data
X_test = np.array([[2000], [1850], [1300], [2200], [1600]])

# Predicting prices using the model
y_pred = model.predict(X_test)

# Printing predicted prices
for i in range(len(X_test)):
    print("Predicted price for a house with {} square feet: ${}".format(X_test[i][0], round(y_pred[i], 2)))
