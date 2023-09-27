import re

def extract_dag_name(file_path):
    """
    Extracts the DAG name from a Python file containing an Airflow DAG definition.

    Parameters:
    file_path (str): Path to the Python file.

    Returns:
    str: The extracted DAG name or None if not found.
    """
    dag_name_pattern = r"DAG\s*\(\s*'(\w+)'\s*,"
    with open(file_path, 'r') as file:
        file_contents = file.read()
        match = re.search(dag_name_pattern, file_contents)
        if match:
            return match.group(1)
    return None

# Example usage
file_path = 'dags/2023-09-22_decision tree.py'
dag_name = extract_dag_name(file_path)
print('Extracted DAG name:', dag_name)
