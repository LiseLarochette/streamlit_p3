import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Charger les datasets
datasets = {
    'Maladie 1': 'path/to/dataset1.csv',
    'Maladie 2': 'path/to/dataset2.csv',
    'Maladie 3': 'path/to/dataset3.csv',
    'Maladie 4': 'path/to/dataset4.csv',
    'Maladie 5': 'path/to/dataset5.csv'
}

# Fonction pour charger et préparer les données
def load_data(path):
    data = pd.read_csv(path)
    X = data.drop(columns='target')
    y = data['target']
    return X, y

# Initialiser les modèles pour chaque maladie
models = {}
for disease, path in datasets.items():
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression())])

    model.fit(X_train, y_train)
    models[disease] = model

# Créer l'interface utilisateur avec Streamlit
st.title("Prédiction de Maladies")

# Sélectionner la maladie à prédire
disease_choice = st.selectbox("Choisissez une maladie à prédire", list(datasets.keys()))

# Afficher les variables à entrer pour la maladie sélectionnée
X_example, _ = load_data(datasets[disease_choice])
inputs = {}
for col in X_example.columns:
    if X_example[col].dtype == 'float64' or X_example[col].dtype == 'int64':
        inputs[col] = st.number_input(f"Entrez la valeur pour {col}", value=float(X_example[col].mean()))

# Préparer les données pour la prédiction
input_data = pd.DataFrame([inputs])

# Faire la prédiction lorsque le bouton est cliqué
if st.button("Prédire"):
    model = models[disease_choice]
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write(f"Le patient a {disease_choice}.")
    else:
        st.write(f"Le patient n'a pas {disease_choice}.")
