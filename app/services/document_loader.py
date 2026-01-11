from typing import List
from pathlib import Path
import pandas as pd
from langchain.schema import Document

def load_data_for_role(role: str) -> List[Document]:
    """
    Load data for a specific role and return LangChain Documents.
    """
    data_path = Path("resources/data") / role
    
    if not data_path.exists():
        print(f"Warning: Data path does not exist for role: {role}")
        return []

    documents = []

    for filepath in data_path.iterdir():
        if filepath.is_file():
            try:
                if filepath.suffix.lower() == ".md":
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": filepath.name,
                                "role": role.lower(),
                                "type": "md"
                            }
                        )
                    )

                # elif filepath.suffix.lower() == ".csv":
                #     df = pd.read_csv(filepath)
                #     rows = df.to_dict(orient="records")
                    
                #     for row in rows:
                #         content = "\n".join(f"{k}: {v}" for k, v in row.items() if pd.notna(v))
                #         documents.append(
                #             Document(
                #                 page_content=content,
                #                 metadata={
                #                     "source": filepath.name,
                #                     "role": role.lower(),
                #                     "type": "csv"
                #                 }
                #             )
                #         )
                #     print(f"Loaded CSV file: {filepath.name} ({len(rows)} rows)")

                else:
                    print(f"Skipping unsupported file type: {filepath.name}")

            except Exception as e:
                print(f"Error reading file {filepath}: {e}")

    return documents