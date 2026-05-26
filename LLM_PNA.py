# --- Code Cell ---
# Configuration

import spacy
from spacy.language import Language
import requests
import pymysql
#import subprocess
import sys
#import ollama
import os
from path import Path
import medspacy
from medspacy_pna import build_nlp
from medspacy_pna.document_classification.radiology_document_classifier import TIER_2_CLASSES, ALTERNATE_DIAGNOSES


def is_ollama_ready() -> bool:
    try:
        response = requests.get("http://localhost:11434")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def send_prompt_to_ollama(prompt: str) -> str:
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "llama3"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

def save_result_to_file(piv: int, accession: str,  question: str, response: str, output_file):
    a = response.split('\n\n')
    formatted_output = (f'{str(piv)}|{accession}|{question}|{a[0]}|{a[1]}\n')
    fout = Path(output_file)
    with fout.open("a", encoding="utf-8") as f:
        f.write(formatted_output)


def get_blob_for_client(db_connection, patientvisitid, accession):
    """
    Retrieves the blob text for a given client using both patientvisitid and accession.
    Assumes the table 'clients' has columns 'patientvisitid', 'accession', and 'blob'.
    """
    cursor = db_connection.cursor()
    query = "SELECT note_text FROM sspSB.PNA2_100_radnotes WHERE patientvisitid = %s AND accession = %s"
    cursor.execute(query, (patientvisitid, accession))
    row = cursor.fetchone()
    return row[0] if row else None



# def run_ollama(prompt, model="llama3"):
#     """
#     Runs the Ollama command-line tool with the specified prompt.
#     Adjust the command as necessary depending on your installation.
#     """
#     try:
#         result = subprocess.run(
#             ["ollama", "run", model, "--prompt", prompt],
#             capture_output=True,
#             text=True,
#             check=True
#         )
#         return result.stdout.strip()
#     except subprocess.CalledProcessError as e:
#         print(f"Error running Ollama for prompt: {prompt}\n{e}", file=sys.stderr)
#         return None

def db_connect():
    # MySQL database connection parameters
    db_host = "localhost"
    db_user = "cler"
    db_password = "grace000"
    db_name = "sspSB"
    conn = pymysql.connect(host=db_host,
                             user=db_user,
                             password=db_password,
                             database=db_name, port=3306, local_infile=1)
    return conn

@Language.factory("medspacy_concept_tagger")
class DummyConceptTagger:
    def __init__(self, nlp, name):
        """No special initialization needed."""
    def __call__(self, doc):
        # This component does nothing and just returns the Doc
        return doc
    def add(self, rules):
        # The MedSpaCy pipeline will try to call .add(...) to add concept tagger rules.
        # We override this to a no-op to safely ignore those rules.
        return None
    
def main():
    # File paths (adjust these paths as needed)
    srcdir = "c:/Work/Data/Textwork"
    questions_file = os.path.join(srcdir,"questionsPNA.txt")
    output_file =  os.path.join(srcdir,"responsesPNA.txt")
    nlp_radiology = build_nlp(domain="radiology")
    
       
    # Read questions from file (one per line)
    with open(questions_file, "r") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Connect to the MySQL database
    try:
        conn = db_connect()
    except pymysql.Error as err:
        print(f"Error connecting to MySQL: {err}", file=sys.stderr)
        sys.exit(1)
           
# Retrieve the list of clients from a table containing patientvisitid and accession.
    try:
        client_cursor = conn.cursor()
        client_cursor.execute("SELECT distinct patientvisitid, accession FROM sspSB.small_notes")
        client_records = client_cursor.fetchall()
    except pymysql.connector.Error as err:
        print(f"Error querying client list: {err}", file=sys.stderr)
        conn.close()
        sys.exit(1)
    
    # Read questions from file (one per line)
    try:
        with open(questions_file, "r") as f:
            questions = [line.strip() for line in f if line.strip()]
    except Exception as err:
        print(f"Error reading questions file: {err}", file=sys.stderr)
        conn.close()
        sys.exit(1)
    
    # Open CSV file for writing the results
    with open(output_file, "w", newline="") as fout:
        header = 'PatientVisitID|Accession|Question|Adjudication|Reasoning\n'
        fout.write(header)
        
        # Process each client from the client_records
        for patientvisitid, accession in client_records:
            print(f"Processing patientvisitid: {patientvisitid}, accession: {accession}")
            blob_text = get_blob_for_client(conn, patientvisitid, accession)
            if blob_text is None:
                print(f"No blob found for patientvisitid: {patientvisitid}, accession: {accession}", file=sys.stderr)
                continue
            result = nlp_radiology(blob_text)
            print("Medspacy - Is pneumonia present?", result._.document_classification)
            
            # For each question, build the prompt and query the LLM
            for question in questions:
                # Concatenate the question and the blob text.
                prompt = f'{question}:"{blob_text}"'
                response = send_prompt_to_ollama(prompt)
                if response is None:
                    response = "Error or no response"
                
#                print(response)
                
                # Write the patientvisitid, accession, question, and LLM response to the CSV file.
                save_result_to_file(patientvisitid, accession, question, response, output_file)
    
    conn.close()
    print(f"All results written to {output_file}")

if __name__ == "__main__":
    main()
    
    
    
