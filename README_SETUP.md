# AI Clinical Decision Support — Type 2 Diabetes

## Requirements

- Python 3.12+ recommended
- Internet connection for installing packages and downloading the embedding model
- A Gemini API key

## Setup on a new computer

1. Clone/download the project and open a terminal in the project folder.
2. Create a virtual environment:

```powershell
python -m venv venv
```

3. Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can skip activation and use the venv Python directly.

4. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

5. Create a `.env` file in the project root:

```env
GEMINI_API_KEY=PUT_YOUR_GEMINI_KEY_HERE
```

Do NOT upload `.env` to GitHub.

6. Make sure the project contains:

```text
data/
  guideline_1.pdf
  guideline_2.pdf

src/
  ingest.py
  vector_store.py
  retrieve.py
  generate.py
  evaluate.py
  test_gemini.py
```

7. If `chroma_db/` is not included in the repository, build it first:

```powershell
python .\src\vector_store.py
```

8. Run the generation layer:

```powershell
python .\src\generate.py
```

## Important

Each person running the project needs their own Gemini API key.

The embedding model may be downloaded the first time the project runs, so the first run can take longer and requires internet access.
