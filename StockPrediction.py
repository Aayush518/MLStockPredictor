import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import math

# Load the data into a DataFrame
df = pd.read_csv("all_stocks_5yr.csv")

df = df.loc[df['Name'] == 'CSCO']

# Preprocess the data
df.fillna(value=-99999, inplace=True)
forecast_col = 'close'
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label', 'Name', 'date'], axis=1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Train the linear regression model
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("Model confidence:", confidence)

# Make predictions
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan
last_date = df.iloc[-1]['date']
last_date = dt.datetime.strptime(last_date, '%Y-%m-%d').timestamp()
last_unix = last_date
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = dt.datetime.fromtimestamp(next_unix).strftime('%Y-%m-%d')
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# Plot the data
style.use('ggplot')
df['close'].plot()
df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='best')
plt.show()
