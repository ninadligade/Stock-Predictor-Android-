import numpy as np
import math
import datetime
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from iexfinance import get_historical_data
import matplotlib.pyplot as plt2

COMPANY_LIST = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']


def get_stock_data(stock_name="AAPL", normalized=0):
    df = get_historical_data(stock_name, start="2015-10-10", end="2017-10-10",
                             output_format='pandas')  # pd.DataFrame(stocks)
    # print(df)
    df.drop(df.columns[[2, 4]], axis=1, inplace=True)
    return df


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()  # pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]


def build_model2(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    stock_name = "AAPL"
    df = get_stock_data(stock_name)
    # print(df.tail())
    today = datetime.date.today()
    file_name = stock_name + '_stock_%s.csv' % today
    df.to_csv(file_name)
    df['high'] = df['high'] / 1000
    df['open'] = df['open'] / 1000
    df['close'] = df['close'] / 1000
    # print(df.head(5))
    window = 5
    X_train, y_train, X_test, y_test = load_data(df[::-1], window)
    # print("X_train", X_train.shape)
    # print("y_train", y_train.shape)
    # print("X_test", X_test.shape)
    # print("y_test", y_test.shape)
    model = build_model2([3, 5, 1])
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        epochs=500,
        validation_split=0.1,
        verbose=0)
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    diff = []
    ratio = []
    p = model.predict(X_test)
    for u in range(len(y_test)):
        pr = p[u][0]
        ratio.append((y_test[u] / pr) - 1)
        diff.append(abs(y_test[u] - pr))
        # print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))
    print(p)
    # plt2.plot(p, color='red', label='prediction')
    # plt2.plot(y_test, color='blue', label='y_test')
    # plt2.legend(loc='upper left')
    # plt2.show()