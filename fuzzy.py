import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

# Load dataset from CSV file
def load_disease_data(csv_file):
    return pd.read_csv(csv_file)

# Define fuzzy variables for symptoms
fever = ctrl.Antecedent(np.arange(36, 42, 1), 'fever')  # Temperature in Celsius
cough = ctrl.Antecedent(np.arange(0, 11, 1), 'cough')  # Cough severity scale 0-10
fatigue = ctrl.Antecedent(np.arange(0, 11, 1), 'fatigue')  # Fatigue severity scale 0-10

# Define fuzzy variable for disease
disease = ctrl.Consequent(np.arange(0, 11, 1), 'disease')

# Define fuzzy membership functions for symptoms
fever['mild'] = fuzz.trimf(fever.universe, [36, 37, 38])
fever['moderate'] = fuzz.trimf(fever.universe, [37, 38, 39])
fever['severe'] = fuzz.trimf(fever.universe, [38, 40, 41])

cough['mild'] = fuzz.trimf(cough.universe, [0, 2, 4])
cough['moderate'] = fuzz.trimf(cough.universe, [3, 5, 7])
cough['severe'] = fuzz.trimf(cough.universe, [6, 8, 10])

fatigue['mild'] = fuzz.trimf(fatigue.universe, [0, 2, 4])
fatigue['moderate'] = fuzz.trimf(fatigue.universe, [3, 5, 7])
fatigue['severe'] = fuzz.trimf(fatigue.universe, [6, 8, 10])

# Define fuzzy membership functions for diseases
disease['flu'] = fuzz.trimf(disease.universe, [0, 2, 4])
disease['cold'] = fuzz.trimf(disease.universe, [3, 4, 5])
disease['malaria'] = fuzz.trimf(disease.universe, [5, 6, 7])
disease['diabetes'] = fuzz.trimf(disease.universe, [7, 8, 9])
disease['hypertension'] = fuzz.trimf(disease.universe, [8, 9, 10])

# Define fuzzy rules
rule1 = ctrl.Rule(fever['severe'] & cough['moderate'] & fatigue['severe'], disease['flu'])
rule2 = ctrl.Rule(cough['moderate'] & cough['severe'], disease['cold'])
rule3 = ctrl.Rule(fever['moderate'] & fatigue['severe'], disease['malaria'])
rule4 = ctrl.Rule(fatigue['moderate'] & fever['moderate'], disease['diabetes'])
rule5 = ctrl.Rule(fatigue['severe'] & fever['mild'], disease['hypertension'])

# Build control system
disease_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
disease_predictor = ctrl.ControlSystemSimulation(disease_ctrl)

# Function to predict disease and recommend medicine
def predict_disease_and_medicine(fever_input, cough_input, fatigue_input, disease_data):
    # Input patient symptoms
    disease_predictor.input['fever'] = fever_input
    disease_predictor.input['cough'] = cough_input
    disease_predictor.input['fatigue'] = fatigue_input

    # Perform fuzzy calculation
    disease_predictor.compute()
    disease_output = disease_predictor.output['disease']

    # Determine predicted disease based on fuzzy output
    if disease_output < 2:
        predicted_disease = "Flu"
    elif 2 <= disease_output < 4:
        predicted_disease = "Cold"
    elif 4 <= disease_output < 6:
        predicted_disease = "Malaria"
    elif 6 <= disease_output < 8:
        predicted_disease = "Diabetes"
    elif disease_output >= 8:
        predicted_disease = "Hypertension"
    else:
        predicted_disease = "Unknown"

    # Find recommended medicine from dataset
    medicine = disease_data[disease_data['Disease'] == predicted_disease]['Recommended Medicine'].values[0]

    return predicted_disease, medicine

# Load the dataset
disease_data = load_disease_data('disease_dataset.csv')

# Input patient symptoms
fever_input = 39  # Example input
cough_input = 6   # Example input
fatigue_input = 7  # Example input

# Predict the disease and recommend medicine
predicted_disease, recommended_medicine = predict_disease_and_medicine(fever_input, cough_input, fatigue_input, disease_data)

# Output the result
print(f"Predicted Disease: {predicted_disease}")
print(f"Recommended Medicine: {recommended_medicine}")
