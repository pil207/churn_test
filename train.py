import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib 

# Charger les données                                
data = pd.read_csv('data/train_data.csv') 


X = data[['Age', 'Account_Manager', 'Num_Sites', 'Years']] 

# Sélectionner la colonne cible
y = data['Churn']

# Créer et entraîner le modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle sur les données
model.fit(X, y)

# Sauvegarder le modèle entraîné avec joblib (sans dépendances pandas)
joblib.dump(model, 'data/churn_model_clean.pkl')

print("Modèle de régression logistique entraîné et sauvegardé sous 'churn_model_clean.pkl'.")
