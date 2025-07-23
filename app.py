from flask import Flask, request, jsonify, send_file, render_template, Response
import pandas as pd
import os
import matplotlib.pyplot as plt
import chardet
import re
import unicodedata
from io import BytesIO, StringIO
import html
from dateutil import parser
import numpy as np
from word2number import w2n

app = Flask(__name__)
uploaded_df = pd.DataFrame()
history_stack = []

def normalize_unicode(df):
    return df.applymap(lambda x: unicodedata.normalize('NFKC', str(x)) if isinstance(x, str) else x)

def save_history():
    global uploaded_df, history_stack
    history_stack.append(uploaded_df.copy())

@app.route('/export_csv')
def export_csv():
    global uploaded_df
    if uploaded_df.empty:
        return "No data to export.", 400

    csv_buffer = StringIO()
    uploaded_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return Response(
        csv_buffer.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=cleaned_data.csv'}
    )

@app.route('/export_excel')
def export_excel():
    global uploaded_df
    if uploaded_df.empty:
        return "No data to export.", 400

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        uploaded_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
    excel_buffer.seek(0)

    return send_file(
        excel_buffer,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='cleaned_data.xlsx'
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_csv', methods=['POST'])
def load_csv():
    global uploaded_df, history_stack
    file = request.files.get('file')
    if file is None:
        return "No file uploaded.", 400
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    file.seek(0)
    uploaded_df = pd.read_csv(file, encoding=encoding)
    history_stack.clear()  # Clear history when loading new data
    return uploaded_df.to_html()

@app.route('/remove_special_chars')
def remove_special_chars():
    global uploaded_df
    save_history()
    def clean_special_chars(val):
        if isinstance(val, str):
            return re.sub(r"[^A-Za-z0-9\s\-',&]", '', val)
        return val

    uploaded_df = uploaded_df.applymap(clean_special_chars)
    return "<p><b>Removed special characters (except - ' ,) from text columns.</b></p>" + uploaded_df.to_html()

@app.route('/load_sample')
def load_sample():
    global uploaded_df, history_stack
    uploaded_df = pd.DataFrame({
        'Name': [
            '  john', 'Jane  ', 'JANE', 'John', 'Alice', 'alice ', ' Bob', 'bob', 'BOB', None,
            'Charlie', ' charlie ', 'Dana', 'dana', 'D@n@', 'Eve', 'eve', 'EV3', 'Mallory', ' mallory ',
            'Zoe', 'Zoe', 'ZOE ', '', 'Grace', ' grace', 'Oscar', 'Oscar', 'oscar', 'Oscar'
        ],
        'Age': [
            '25', 30, 'thirty', 25, None, 27, 31, '31', 31, 29,
            'NaN', 33, 45, 45, 38, 28, '28', None, 'thirty-two', 32,
            24, 24, '24 ', None, 'N/A', 36, 40, None, 40, 40
        ],
        'Gender': [
            'M', 'F', 'f', 'm', 'F', 'F', 'M', 'm', 'M', 'M',
            None, 'M', 'F', 'F', 'f', 'F', 'f', 'F', 'F', None,
            'Female', 'female', 'F ', 'M', '', 'F', 'Male', 'male', 'm', 'M'
        ],
        'Email': [
            'john@example.com', ' JANE@MAIL.COM ', 'jane@mail', 'john@', 'alice@email.com', 'alice@EMAIL.COM',
            'bob@sample.net', 'bob@@sample.com', 'bob@gmail.com', 'invalid-email',
            '', 'charlie@mail.com', 'dana@mail.com', 'dana@MAIL.COM', 'd@n@!mail.com',
            'eve@mail.com', 'EVE@MAIL.com', 'ev3@mail', 'mallory@mail.com', ' mallory@web.com ',
            'zoe@gmail', 'zoe@', 'zoe@example.com', None, 'grace@mail', ' grace@web.com', 'oscar@site', '', 'oscar@site.com', 'OSCAR@site.com'
        ],
        'JoinDate': [
            '2020-01-15', '01/15/2020', '15-Jan-2020', '2020/01/15', 'Jan 15 2020', '2020.01.15',
            '20200115', '15/01/2020', '2020-13-01', 'Not a date',
            '', '2020-01-15', '2021-02-30', '2022/02/28', 'Feb 28, 2022',
            '03-05-2021', '2021/05/03', 'May 3rd, 2021', None, '2021-05-03',
            '2021-05-03', '2021-5-3', '03 May 2021', 'N/A', 'null', '2022-06-15', '2022-06-15', '', '2022-06-15', '2022-06-15'
        ],
        'Salary': [
            '50000', '60,000', '50K', '55000.50', '$58000', None,
            '59000', 'invalid', '', '52000',
            '48000', 'NaN', '61000', '62000', '63,000',
            '64000', '65000', '66000', '67000', 'not available',
            '68000', '68000.00', '69000', '70000', '70000', '71000', '72000', '0', None, '73000'
        ],
        'Active': [
            'Yes', 'No', 'yes', 'no', 'TRUE', 'FALSE',
            'true', 'false', 1, 0,
            'Y', 'N', '', '1', '0',
            'Active', 'Inactive', None, 'yes', 'no',
            'enabled', 'disabled', 'Yes', 'No', 'n/a', 'True', 'False', '', '1', '0'
        ]
    })
    history_stack.clear()
    return uploaded_df.to_html()


@app.route('/remove_duplicates_by_column')
def remove_duplicates_by_column():
    global uploaded_df
    column = request.args.get('column')

    if not column:
        return "Error: Column parameter is required.", 400
    if column not in uploaded_df.columns:
        return f"Error: Column '{column}' does not exist in the data.", 400

    save_history()
    before = len(uploaded_df)
    uploaded_df = uploaded_df.drop_duplicates(subset=[column])
    after = len(uploaded_df)
    affected = before - after
    return f"<p><b>Removed {affected} duplicate rows based on column '{column}'.</b></p>" + uploaded_df.to_html()

@app.route('/standardize_text')
def standardize_text():
    global uploaded_df
    save_history()
    uploaded_df = uploaded_df.applymap(lambda x: x.title() if isinstance(x, str) else x)
    return uploaded_df.to_html()

@app.route('/missing_values')
def missing_values():
    global uploaded_df
    missing_rows = uploaded_df[uploaded_df.isnull().any(axis=1)]
    return missing_rows.to_json(orient='records')

@app.route('/drop_missing')
def drop_missing():
    global uploaded_df
    save_history()
    before = len(uploaded_df)
    uploaded_df = uploaded_df.dropna()
    after = len(uploaded_df)
    affected = before - after
    return f"<p><b>Dropped {affected} rows with missing values.</b></p>" + uploaded_df.to_html()

@app.route('/fill_missing')
def fill_missing():
    global uploaded_df
    save_history()
    before_missing = uploaded_df.isnull().sum().sum()
    uploaded_df = uploaded_df.fillna(0)
    after_missing = uploaded_df.isnull().sum().sum()
    filled = before_missing - after_missing
    return f"<p><b>Filled {filled} missing cells with 0.</b></p>" + uploaded_df.to_html()

@app.route('/sort_values')
def sort_values():
    global uploaded_df
    column = request.args.get('column')

    if column is None or column not in uploaded_df.columns.astype(str).tolist():
        return f"Invalid column name: {html.escape(str(column))}", 400

    save_history()
    uploaded_df.sort_values(by=column, inplace=True)
    return f"<p><b>Sorted by '{html.escape(str(column))}'.</b></p>" + uploaded_df.to_html()

@app.route('/sort_interface')
def sort_interface():
    global uploaded_df
    if uploaded_df.empty:
        return "No data loaded.", 400

    uploaded_df.columns = uploaded_df.columns.map(lambda x: str(x) if x is not None else "Unnamed")
    options_html = ''.join(
        f'<option value="{html.escape(col)}">{html.escape(col)}</option>'
        for col in uploaded_df.columns
    )

    return f'''
        <form action="/sort_values" method="get">
            <label for="column">Sort by column:</label>
            <select name="column" id="column">
                {options_html}
            </select>
            <button type="submit">Sort</button>
        </form>
        <br>
        {uploaded_df.to_html()}
    '''

@app.route('/bar_chart')
def bar_chart():
    global uploaded_df
    plt.figure()
    uploaded_df[uploaded_df.columns[0]].value_counts().plot(kind='bar')
    plt.title('Bar Chart')
    path = os.path.join('static', 'bar_chart.png')
    plt.savefig(path)
    plt.close()
    return "Bar chart generated."

@app.route('/pie_chart')
def pie_chart():
    global uploaded_df
    plt.figure()
    uploaded_df[uploaded_df.columns[0]].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Pie Chart')
    path = os.path.join('static', 'pie_chart.png')
    plt.savefig(path)
    plt.close()
    return "Pie chart generated."

@app.route('/change_dtype')
def change_dtype():
    global uploaded_df
    column = request.args.get('column')
    dtype = request.args.get('dtype')

    if column not in uploaded_df.columns:
        return f"Invalid column: {column}", 400

    save_history()
    try:
        uploaded_df[column] = uploaded_df[column].astype(dtype)
    except Exception as e:
        return f"Conversion failed: {e}", 400

    return f"<p><b>Changed '{column}' to {dtype}.</b></p>" + uploaded_df.to_html()

@app.route('/replace_values')
def replace_values():
    global uploaded_df
    column = request.args.get('column')
    to_replace = request.args.get('replace_from')
    new_value = request.args.get('to')

    if column not in uploaded_df.columns:
        return f"Invalid column: {column}", 400

    save_history()

    # Replace only if full value matches (case-insensitive)
    uploaded_df[column] = uploaded_df[column].apply(
        lambda x: new_value if str(x).strip().lower() == to_replace.lower() else x
    )

    return f"<p><b>Replaced '{to_replace}' with '{new_value}' in '{column}' (case-insensitive).</b></p>" + uploaded_df.to_html()




@app.route('/rename_column')
def rename_column():
    global uploaded_df
    old_name = request.args.get('old')
    new_name = request.args.get('new')

    if old_name not in uploaded_df.columns:
        return f"Column '{old_name}' not found.", 400

    save_history()
    uploaded_df.rename(columns={old_name: new_name}, inplace=True)
    return f"<p><b>Renamed '{old_name}' to '{new_name}'.</b></p>" + uploaded_df.to_html()

@app.route('/undo_replace')
def undo_replace():
    global uploaded_df, history_stack
    if history_stack:
        uploaded_df = history_stack.pop()
        return "<p><b>Undo successful.</b></p>" + uploaded_df.to_html()
    else:
        return "Nothing to undo.", 400

@app.route('/drop_column')
def drop_column():
    global uploaded_df
    column = request.args.get('column')

    if column not in uploaded_df.columns:
        return f"Column '{column}' not found.", 400

    save_history()
    uploaded_df.drop(columns=[column], inplace=True)
    return f"<p><b>Dropped column '{column}'.</b></p>" + uploaded_df.to_html()

@app.route('/drop_value')
def drop_value():
    global uploaded_df
    column = request.args.get('column')
    value = request.args.get('value')

    if column not in uploaded_df.columns:
        return f"Column '{column}' not found.", 400

    save_history()
    before = len(uploaded_df)
    uploaded_df = uploaded_df[~uploaded_df[column].astype(str).str.contains(value, na=False)]
    after = len(uploaded_df)
    affected = before - after

    return f"<p><b>Dropped {affected} rows where '{column}' contains '{value}'.</b></p>" + uploaded_df.to_html()

@app.route('/data_profile')
def data_profile():
    global uploaded_df
    if uploaded_df.empty:
        return "No data loaded.", 400

    profile = {}
    profile['Total Records'] = len(uploaded_df)

    for col in uploaded_df.columns:
        col_lower = col.lower()
        series = uploaded_df[col]

        # Missing values
        missing_count = series.isnull().sum()
        if missing_count > 0:
            profile[f"Missing values in '{col}'"] = missing_count

        # Invalid phone numbers
        if 'phone' in col_lower:
            pattern = r'^\+?\d{10,15}$'
            invalid_phones = series.apply(lambda x: not bool(re.fullmatch(pattern, str(x))) if pd.notnull(x) else False).sum()
            profile[f"Invalid Phone Numbers in '{col}'"] = invalid_phones

        # Invalid emails
        if 'email' in col_lower:
            pattern = r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
            invalid_emails = series.apply(lambda x: not bool(re.fullmatch(pattern, str(x))) if pd.notnull(x) else False).sum()
            profile[f"Invalid Emails in '{col}'"] = invalid_emails

        # Non-standard country codes
        if 'country' in col_lower:
            nonstandard = series.apply(lambda x: not bool(re.fullmatch(r'^[A-Za-z]{2,3}$', str(x))) if pd.notnull(x) else False).sum()
            profile[f"Non-standard Country Codes in '{col}'"] = nonstandard

        # Missing IDs
        if 'id' in col_lower:
            missing_ids = series.isnull().sum()
            profile[f"Missing IDs in '{col}'"] = missing_ids

        # Duplicates per column (excluding nulls)
        duplicate_vals = series[series.notnull()].duplicated().sum()
        if duplicate_vals > 0:
            profile[f"Duplicate Values in '{col}'"] = duplicate_vals

    # Row-level duplicate detection
    profile['Duplicate Rows (Full Match)'] = uploaded_df.duplicated().sum()

    # Convert to HTML table
    html_output = "<h3>Input Data Profile (Pre-Cleaning)</h3><table border='1'><tr><th>Metric</th><th>Value</th></tr>"
    for key, value in profile.items():
        html_output += f"<tr><td>{key}</td><td>{value}</td></tr>"
    html_output += "</table>"

    return html_output


@app.route('/find')
def find():
    global uploaded_df
    keyword = request.args.get('keyword')
    column = request.args.get('column')

    if not keyword:
        return "Please provide a keyword using ?keyword=your_value", 400

    if uploaded_df.empty:
        return "No data loaded.", 400

    if column:
        if column not in uploaded_df.columns:
            return f"Column '{column}' not found.", 400
        mask = uploaded_df[column].astype(str).str.contains(keyword, case=False, na=False)
        filtered_df = uploaded_df[mask]
    else:
        mask = uploaded_df.apply(lambda row: row.astype(str).str.contains(keyword, case=False, na=False).any(), axis=1)
        filtered_df = uploaded_df[mask]

    if filtered_df.empty:
        return f"No matching records found for '{keyword}'."

    return f"<p><b>Found {len(filtered_df)} matching rows for keyword '{keyword}'</b></p>" + filtered_df.to_html()

from word2number import w2n

@app.route('/magic_clean')
def magic_clean():
    global uploaded_df
    save_history()

    def clean_column_names(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
        return df

    def clean_value(val, col_name):
        if pd.isnull(val):
            return None

        if isinstance(val, str):
            original_val = val.strip()
            val = original_val  # keep original as backup

            # Clean email fields
            if 'email' in col_name:
                val = re.sub(r'@+', '@', val)
                if '@' in val and not re.search(r'\.\w+$', val):
                    val += '.com'
                if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', val):
                    return val.lower()
                else:
                    return None

            # Clean boolean-like values
            if val.lower() in ['yes', 'true', 'y', '1', 'active', 'enabled']:
                return 1
            elif val.lower() in ['no', 'false', 'n', '0', 'inactive', 'disabled']:
                return 0

            # Try to convert word numbers
            try:
                word_to_num = w2n.word_to_num(val.lower())
                return word_to_num
            except:
                pass

            # Clean numeric values (with k/m symbols, $, commas)
            val_numeric = val.lower().replace(',', '').replace('$', '')
            if val_numeric.endswith('k'):
                try:
                    return int(float(val_numeric.replace('k', '')) * 1000)
                except:
                    pass
            elif val_numeric.endswith('m'):
                try:
                    return int(float(val_numeric.replace('m', '')) * 1_000_000)
                except:
                    pass
            elif re.fullmatch(r'-?\d+(\.\d+)?', val_numeric):
                return float(val_numeric) if '.' in val_numeric else int(val_numeric)

            # Try parsing as a date
            try:
                parsed = parser.parse(val, fuzzy=True, dayfirst=False)
                return parsed.strftime('%Y-%m-%d')
            except:
                pass

            # If column name likely refers to a name or title, standardize it
            if any(key in col_name for key in ['name', 'title', 'position']):
                val = re.sub(r'[^A-Za-zÀ-ÿ\'\- ]+', '', val)
                return val.title().strip()

            # For other text columns, keep original (may contain valid punctuation/numbers)
            return original_val

        # Convert Python boolean to 1/0
        if isinstance(val, bool):
            return int(val)

        return val

    uploaded_df = clean_column_names(uploaded_df)
    uploaded_df = uploaded_df.apply(lambda col: col.apply(lambda val: clean_value(val, col.name)))
    uploaded_df.drop_duplicates(inplace=True)

    return "<p><b>✨ Magic Clean applied!</b></p>" + uploaded_df.to_html()

@app.route('/generate_chart_data')
def generate_chart_data():
    global uploaded_df
    x_col = request.args.get('x')
    y_col = request.args.get('y')

    if uploaded_df.empty or x_col not in uploaded_df.columns or y_col not in uploaded_df.columns:
        return jsonify({'labels': [], 'values': []})

    # Drop missing or non-numeric
    df = uploaded_df[[x_col, y_col]].dropna()
    try:
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna()
    except:
        pass

    grouped = df.groupby(x_col)[y_col].sum().reset_index()
    labels = grouped[x_col].astype(str).tolist()
    values = grouped[y_col].tolist()

    return jsonify({'labels': labels, 'values': values})

@app.route('/upload_mapping_csv', methods=['POST'])
def upload_mapping_csv():
    global mapping_df
    file = request.files.get('file')
    if file is None:
        return "No mapping file uploaded.", 400
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    file.seek(0)
    mapping_df = pd.read_csv(file, encoding=encoding)
    return f"<p><b>Mapping CSV loaded successfully with {len(mapping_df)} rows.</b></p>" + mapping_df.to_html()

@app.route('/apply_subviolation_mapping')
def apply_subviolation_mapping():
    global uploaded_df, mapping_df

    if uploaded_df.empty or mapping_df.empty:
        return "Main data or mapping data is not loaded.", 400

    target_col = request.args.get("target_col")
    if not target_col:
        return "Missing 'target_col' query parameter.", 400

    if target_col not in uploaded_df.columns:
        return f"Column '{target_col}' does not exist in uploaded data.", 400

    if 'id' not in mapping_df.columns or 'name' not in mapping_df.columns:
        return "Mapping file must have 'id' and 'name' columns.", 400

    save_history()

    uploaded_df[target_col] = uploaded_df[target_col].map(
        dict(zip(mapping_df['id'], mapping_df['name']))
    )

    return f"<p><b>Applied mapping on column <code>{target_col}</code>.</b></p>" + uploaded_df.to_html()

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host="127.0.0.1", port=5000, debug=True)
