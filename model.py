import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("C:\Users\chikk\OneDrive\文档\Flask_beerserving\beer-servings.csv")


df = df.dropna(subset=['total_litres_of_pure_alcohol', 'beer_servings', 'spirit_servings', 'wine_servings', 'country', 'continent'])

# Features and target variable
X = df[['beer_servings', 'spirit_servings', 'wine_servings', 'continent']]
y = df['total_litres_of_pure_alcohol']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=['continent'])


X_train, X_test, y_train, y_test=train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


with open('total_alcohol_model.pkl', 'wb') as file:
    pickle.dump((model, X_encoded.columns), file)

print("Model trained and saved successfully!")
