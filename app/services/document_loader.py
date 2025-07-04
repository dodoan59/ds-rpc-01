import os
import pandas as pd
from typing import Dict, List, Any

# Return list of dictionaries containing 'content' and 'type'
def load_data_for_role(role: str) -> List[Dict[str, Any]]:
    """
    Load data for a specific role.
    Returns a list of dictionaries, each containing 'content' and 'type'.
    """
    data_path = os.path.join("resources", "data", role)
    if not os.path.exists(data_path):
        print(f"Warning: Data path does not exist for role: {role}")
        return []

    documents = []
    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        file_extension = filename.split('.')[-1]

        if file_extension == "md":
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    # Add content and type
                    documents.append({"content": f.read(), "type": "md"})
            except Exception as e:
                print(f"Error reading text/markdown file {filepath}: {e}")
        
        elif file_extension == "csv":
            try:
                df = pd.read_csv(filepath)
                for index, row in df.iterrows():
                    row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                    # Add content and type
                    documents.append({"content": row_text, "type": "csv"})
                print(f"Loaded CSV file: {filepath}")
            except Exception as e:
                print(f"Error reading CSV file {filepath}: {e}")
        else:
            print(f"Skipping unsupported file type: {filename}")

    return documents