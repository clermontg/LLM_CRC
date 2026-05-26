"""
LLM_CRC_generic.py

Applies a set of natural-language prompts to clinical text notes stored in a
MySQL database, using a locally-running Ollama LLM to generate answers.

Workflow:
  1. Read a list of questions/prompts from a plain-text file (one per line).
  2. Query the database for a list of (patientvisitid, accession) pairs.
  3. For each pair, retrieve the associated note text from the database.
  4. For each (note, question) combination, build a prompt and send it to Ollama.
  5. Append all (patientvisitid, accession, question, LLM-response) rows to a CSV.
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Ollama service
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3"

# MySQL connection
DB_HOST     = "localhost"
DB_PORT     = 3306
DB_USER     = "cler"
DB_PASSWORD = "xxx"
DB_NAME     = "sspSB"

# File paths
import os
SRC_DIR        = "c:/Work/Data/Textwork"
QUESTIONS_FILE = os.path.join(SRC_DIR, "questionsPNA.txt")
OUTPUT_CSV     = os.path.join(SRC_DIR, "responsesPNA.csv")

# SQL queries
RECORDS_QUERY = "SELECT DISTINCT patientvisitid, accession FROM sspSB.small_notes"
NOTES_QUERY   = (
    "SELECT note_text FROM sspSB.PNA2_100_radnotes "
    "WHERE patientvisitid = %s AND accession = %s"
)

# ---------------------------------------------------------------------------

import csv
import sys

import pymysql
import requests


def is_ollama_ready() -> bool:
    """Return True if the local Ollama service is reachable."""
    try:
        response = requests.get(OLLAMA_BASE_URL)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def send_prompt_to_ollama(prompt: str) -> str:
    """
    Send *prompt* to the Ollama REST API and return the model's text response.

    Returns an error string instead of raising on network failures so that
    processing can continue for remaining records.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"


def db_connect() -> pymysql.connections.Connection:
    """Open and return a pymysql connection using the configured credentials."""
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        local_infile=1,
    )


def get_note_text(db_connection, patientvisitid: int, accession: str):
    """
    Retrieve the note text for a single (patientvisitid, accession) pair.

    Returns the note string, or None if no matching row is found.
    """
    cursor = db_connection.cursor()
    cursor.execute(NOTES_QUERY, (patientvisitid, accession))
    row = cursor.fetchone()
    return row[0] if row else None


def main():
    # -- Read questions --------------------------------------------------
    try:
        with open(QUESTIONS_FILE, "r") as f:
            questions = [line.strip() for line in f if line.strip()]
    except OSError as err:
        print(f"Error reading questions file: {err}", file=sys.stderr)
        sys.exit(1)

    # -- Connect to database ---------------------------------------------
    try:
        conn = db_connect()
    except pymysql.Error as err:
        print(f"Error connecting to MySQL: {err}", file=sys.stderr)
        sys.exit(1)

    # -- Retrieve the list of records to process -------------------------
    try:
        cursor = conn.cursor()
        cursor.execute(RECORDS_QUERY)
        records = cursor.fetchall()
    except pymysql.Error as err:
        print(f"Error querying record list: {err}", file=sys.stderr)
        conn.close()
        sys.exit(1)

    # -- Process each record and write results ---------------------------
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["PatientVisitID", "Accession", "Question", "Response"])

        for patientvisitid, accession in records:
            note_text = get_note_text(conn, patientvisitid, accession)
            if note_text is None:
                print(
                    f"No note found for patientvisitid={patientvisitid}, "
                    f"accession={accession}",
                    file=sys.stderr,
                )
                continue

            for question in questions:
                prompt = f'{question}: "{note_text}"'
                response = send_prompt_to_ollama(prompt)
                writer.writerow([patientvisitid, accession, question, response])

            print(f"Processed patientvisitid={patientvisitid}, accession={accession}")

    conn.close()
    print(f"All results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
