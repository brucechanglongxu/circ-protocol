import sys
import json
from agents.her2_pathway_agent import her2_pathway_agent
from agents.diabetes_care_agent import diabetes_care_agent

def route_agent(patient_path):
    with open(patient_path) as f:
        patient = json.load(f)

    conditions = patient.get("conditions", [])
    icd_codes = [cond["code"] for cond in conditions]

    if any(code.startswith("C50") for code in icd_codes):
        her2_pathway_agent(patient_path)
    elif any(code.startswith("E10") or code.startswith("E11") for code in icd_codes):
        diabetes_care_agent(patient_path)
    else:
        print("No routing rule matched. Please review patient conditions.")
    #if f.read().strip() == "":
    #    print("Error: File is empty.")
    #return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_patient_json>")
        sys.exit(1)
    route_agent(sys.argv[1])
