import json

def her2_pathway_agent(patient_path):
    with open(patient_path) as f:
        patient = json.load(f)
    
    her2_status = next(
        (obs for obs in patient["observations"] if obs["display"].lower().startswith("her2")),
        None
    )
    
    if her2_status and her2_status["value"].lower() == "positive":
        print(f"HER2+ detected for {patient['name'][0]['given'][0]} {patient['name'][0]['family']}")
        print("→ Recommend initiating trastuzumab-based therapy protocol.")
    else:
        print("HER2+ status not confirmed. Clinical decision required.")

if __name__ == "__main__":
    her2_pathway_agent("patients/her2_patient.json")

