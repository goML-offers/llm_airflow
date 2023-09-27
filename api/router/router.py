import sys
from fastapi import APIRouter, HTTPException, Response
from services.DAG_generation import generate_airflow_dag    
import os
from fastapi.responses import FileResponse, JSONResponse
import re
import json
from dotenv import load_dotenv
load_dotenv()


router = APIRouter()
@router.post('/goml/LLM marketplace/LLM Airflow', status_code=201)
def data_generator(prompt: str):
    try:
        if os.path.exists('generated_dag.py'):
            os.remove('generated_dag.py')

        script_file_path = generate_airflow_dag(prompt)
        # Check if the file was saved successfully
        dag_name_pattern = r"DAG\s*\(\s*'(\w+)'\s*,"
        with open(script_file_path, 'r') as file:
            file_contents = file.read()
            match = re.search(dag_name_pattern, file_contents)
            if match:
                script_file_path = match.group(1)
            else:
                script_file_path = None
        
        return {'filename':script_file_path,'filepath':f"http://localhost:8080/dags/{script_file_path}/grid"}
        
    
       
    except Exception as e:
            
            raise HTTPException(status_code=400, detail=str(e))
   
#write a code to upload to s3