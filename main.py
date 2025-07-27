import sys
from agents.her2_pathway_agent import her2_pathway_agent

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_patient_json>")
        sys.exit(1)
    her2_pathway_agent(sys.argv[1])
