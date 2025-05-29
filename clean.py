import csv
import re
import pandas as pd

# 1. Define the input and output file paths
INPUT_CSV = 'USA.csv'
OUTPUT_CSV = 'USA_cleaned.csv'

# 2. Compile the citation‐removal regex
cite_pattern = re.compile(r'\s*\[cite:[^\]]*\]')

# 3. Columns to clean
columns_to_clean = [
    'Test Suite ID',
    'Test Suite Name',
    'Test Case ID',
    'Test Case Name',
    'Prerequisites',
    'Step Number',
    'Step Action',
    'Expected Result'
]

# 4. Load the CSV into a DataFrame
df = pd.read_csv(INPUT_CSV, dtype=str)  # dtype=str preserves everything as text

# 5. Remove citations in each target column
for col in columns_to_clean:
    if col in df.columns:
        df[col] = df[col].fillna('').str.replace(cite_pattern, '', regex=True)

# 6. Save out to Excel
df.to_excel(OUTPUT_XLSX, index=False)

print(f"✅ Cleaned data written to Excel file: {OUTPUT_XLSX}")