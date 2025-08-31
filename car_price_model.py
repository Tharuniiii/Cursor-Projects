# car_price_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_and_save_model():
    # Sample dataset
    data = {
        'year': [2010, 2012, 2015, 2018, 2020],
        'mileage': [70000, 50000, 30000, 20000, 10000],
        'brand': ['Ford', 'BMW', 'Ford', 'BMW', 'Ford'],
        'price': [5000, 12000, 9000, 20000, 25000]
    }
    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=['brand'], drop_first=True)

    X = df.drop('price', axis=1)
    y = df['price']

    model = RandomForestRegressor()
    model.fit(X, y)

    joblib.dump(model, 'car_price_model.pkl')
    joblib.dump(list(X.columns), 'model_columns.pkl')

if __name__ == '__main__':
    train_and_save_model()