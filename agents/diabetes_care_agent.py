import json

def diabetes_care_agent(patient_path):
    with open(patient_path) as f:
        patient = json.load(f)

    name = patient['name'][0]['given'][0] + ' ' + patient['name'][0]['family']
    diagnoses = [d['code'] for d in patient.get("conditions", [])]

    diabetes_icd_codes = {"E10", "E11", "E13"}  # ICD-10 for types of diabetes

    if any(code[:3] in diabetes_icd_codes for code in diagnoses):
        print(f"Diabetes identified for {name}")
        print("→ Recommend initiating A1C monitoring, foot exam, and metformin evaluation.")
    else:
        print(f"No diabetes ICD-10 code found for {name}")
