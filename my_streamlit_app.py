import streamlit as st

# Définir les entrées utilisateur pour les variables médicales
st.title("Prédiction de la maladie hépatique")

age = st.number_input("Age", min_value=0, max_value=100, value=30)
total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, max_value=50.0, value=1.0)
alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, max_value=2000, value=200)
alamine_aminotransferase = st.number_input("Alamine Aminotransferase", min_value=0, max_value=2000, value=25)
albumin_and_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, max_value=3.0, value=1.0)

# Préparer les données pour la prédiction
input_data = pd.DataFrame({
    'Age': [age],
    'Total_Bilirubin': [total_bilirubin],
    'Alkaline_Phosphotase': [alkaline_phosphotase],
    'Alamine_Aminotransferase': [alamine_aminotransferase],
    'Albumin_and_Globulin_Ratio': [albumin_and_globulin_ratio]
})

# Faire la prédiction
if st.button("Prédire"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("Le patient a une maladie hépatique.")
    else:
        st.write("Le patient n'a pas de maladie hépatique.")
