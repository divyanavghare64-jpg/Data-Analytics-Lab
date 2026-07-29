
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df = pd.read_csv('diabetes.csv')

# 2. Separate Features (X) and Target (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Classifier Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. Save the trained model as a .pkl file
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("SUCCESS: 'diabetes_model.pkl' generated successfully!")