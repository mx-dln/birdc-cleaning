✨ Magic Clean (Beta) — Rule-Based Data Cleaning Tool
Magic Clean is a lightweight, Python-based data cleaning web app that helps you auto-clean CSV data with smart, rule-based logic — no Excel wrangling or manual formatting needed. Built with Flask and Pandas, it’s ideal for fast prototyping or integrating into bigger data workflows.

🔧 Note: This is a beta version. OpenAI-powered deep cleaning is coming soon!

🚀 Features

✅ Column Name Normalization

Trims whitespace

Converts to lowercase

Replaces spaces with underscores

Example: Full Name → full_name

✅ Intelligent Cell-Level Cleaning

🧹 Email Fixes: Merges @@ errors, appends .com if missing

🔢 Amount Parsing: Converts $1.2k, 3.5M, etc. to numbers

🔠 Boolean Conversion: Converts yes, active, enabled, etc. → 1

📅 Date Detection: Auto-parses fuzzy dates to YYYY-MM-DD

🔤 Name Formatting: Capitalizes names, removes symbols (e.g., john_doe123!! → John Doe)

💬 Keeps Clean Text: Leaves valid strings untouched

🧠 Word to Number: Converts three thousand five → 3005

✅ De-duplication

Removes exact duplicate rows after cleaning

✅ Safe Undo System

Automatically saves the pre-cleaned state for rollback

🖥️ Live Demo (Coming Soon)
🧪 Example Input → Output
Raw Input	Cleaned Output
YES	1
three thousand	3000
@@user@mail	user@mail.com
July 4th, 2021	2021-07-04
mr. JOHN-doe!!!	Mr. John-Doe

📦 Installation
bash
Copy
Edit
git clone https://github.com/yourusername/magic-clean.git
cd magic-clean
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
python app.py
Visit http://127.0.0.1:5000 to use the interface.

📁 Project Structure
csharp
Copy
Edit
├── app.py               # Flask backend + cleaning logic
├── templates/
│   └── index.html       # UI for CSV upload and results
├── static/
│   └── style.css        # Optional styles
├── requirements.txt
└── README.md
🛠 Tech Stack
Python 3

Flask

Pandas

dateutil, regex, word2number

✅ Progress Indicator for Large Datasets

✅ Undo / Redo History

📄 License
MIT License
