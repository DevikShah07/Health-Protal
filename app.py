from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('final_ml_api3.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a dictionary for the medicine recommendations based on the prognosis
prognosis_medicine_dict = {
'Fungal infection': 'Topical antifungals, oral antifungals',
    'Allergy': 'Antihistamines, decongestants, corticosteroids',
    'GERD': 'Proton pump inhibitors, H2 blockers',
    'Chronic cholestasis': 'Ursodeoxycholic acid, vitamin supplements (A, D, E, K)',
    'Drug Reaction': 'Discontinue offending drug, antihistamines, corticosteroids',
    'Peptic ulcer diseae': 'Proton pump inhibitors, antibiotics for H. pylori infection',
    'AIDS': 'Antiretroviral therapy (ART), immune-boosting supplements',
    'Diabetes ': 'Insulin, oral hypoglycemics',
    'Gastroenteritis': 'Oral rehydration solution, antidiarrheals, antibiotics if bacterial',
    'Bronchial Asthma': 'Inhaled corticosteroids, bronchodilators',
    'Hypertension ': 'ACE inhibitors, beta-blockers, diuretics',
    'Migraine': 'NSAIDs, triptans, anti-nausea medication',
    'Cervical spondylosis': 'Pain relievers, physical therapy, muscle relaxants',
    'Paralysis (brain hemorrhage)': 'Supportive care, physical therapy, anticoagulants if needed',
    'Jaundice': 'Treat underlying cause, avoid hepatotoxic drugs, supportive care',
    'Malaria': 'Antimalarial drugs (e.g., chloroquine, artemisinin-based combination therapy)',
    'Chicken pox': 'Antihistamines for itching, antiviral drugs for severe cases',
    'Dengue': 'Supportive care, fluids, pain relievers (avoid NSAIDs)',
    'Typhoid': 'Antibiotics (e.g., ciprofloxacin, azithromycin)',
    'hepatitis A': 'Supportive care, rest, avoid alcohol',
    'Hepatitis B': 'Antiviral medications, liver support',
    'Hepatitis C': 'Direct-acting antiviral agents, supportive care',
    'Hepatitis D': 'Supportive care, rest',
    'Hepatitis E': 'Supportive care, rest',
    'Alcoholic hepatitis': 'Abstain from alcohol, corticosteroids',
    'Tuberculosis': 'Antibiotics (e.g., isoniazid, rifampicin)',
    'Common Cold': 'Decongestants, antihistamines, NSAIDs',
    'Pneumonia': 'Antibiotics (if bacterial), cough medicine, fever reducers',
    'Dimorphic hemmorhoids(piles)': 'Fiber supplements, topical treatments, surgery if needed',
    'Heart attack': 'Aspirin, thrombolytics, beta-blockers, statins',
    'Varicose veins': 'Compression stockings, sclerotherapy, laser treatment',
    'Hypothyroidism': 'Thyroid hormone replacement',
    'Hyperthyroidism': 'Antithyroid drugs, beta-blockers',
    'Hypoglycemia': 'Glucose tablets, sugary foods/drinks, glucagon injection',
    'Osteoarthristis': 'NSAIDs, physical therapy, joint injections',
    'Arthritis': 'Anti-inflammatory drugs, DMARDs',
    '(vertigo) Paroymsal  Positional Vertigo': 'Antivertigo medications, vestibular therapy',
    'Acne': 'Topical retinoids, benzoyl peroxide, antibiotics',
    'Urinary tract infection': 'Antibiotics (e.g., amoxicillin, ciprofloxacin)',
    'Psoriasis': 'Topical corticosteroids, phototherapy, systemic treatments',
    'Impetigo': 'Topical or oral antibiotics (e.g., mupirocin, amoxicillin)',
    'Malaria' : 'Chloroquine, Artemisinin-based Combination Therapy (ACT) - Artemether-Lumefantrine,     Artesunate-Mefloquine, Primaquine' ,
    'Dengue' : 'Acetaminophen (Paracetamol), Avoid NSAIDs (e.g., Aspirin, Ibuprofen)',
    'Diabetes' : 'If (type-1) - Insulin therapy, If (type- 2) - Metformin, Sulfonylureas, DPP-4 Inhibitors, GLP-1 Agonists, SGLT2 Inhibitors',
    'Chikungunya' : 'Paracetamol, NSAIDs (e.g., Ibuprofen, Naproxen)'
}

def get_medicine_recommendation(disease):
    """Get medicine recommendation based on the disease prediction."""
    return prognosis_medicine_dict.get(disease, "No recommendation available")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    # Collect symptoms from the form data
    symptoms = [
        request.args.get('symptom1'),
        request.args.get('symptom2'),
        request.args.get('symptom3'),
        request.args.get('symptom4'),
        request.args.get('symptom5')
    ]

    # Convert symptoms to feature vector
    input_features = np.array([symptoms_to_features(symptoms)]).reshape(1, -1)

    # Get disease prediction
    disease_prediction = model.predict(input_features)[0]

    # Get medicine recommendation based on disease prediction
    medicine_recommendation = get_medicine_recommendation(disease_prediction)
    
    return render_template('result.html', prediction=disease_prediction, medicine=medicine_recommendation, symptoms=symptoms)

def symptoms_to_features(symptoms):
    all_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze','Urinating_a_lot','heartburn']
    
    # Initialize a binary feature list with 0s for all symptoms
    features = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    
    return features

if __name__ == '__main__':
    app.run(debug=True)
