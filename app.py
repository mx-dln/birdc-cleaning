from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import os
import matplotlib.pyplot as plt
import chardet
import re
import unicodedata
from io import BytesIO, StringIO
from flask import Response
import html


app = Flask(__name__)
uploaded_df = pd.DataFrame()

history_stack = []

def normalize_unicode(df):
    return df.applymap(lambda x: unicodedata.normalize('NFKC', str(x)) if isinstance(x, str) else x)


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
        headers={
            'Content-Disposition': 'attachment; filename=cleaned_data.csv'
        }
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
    global uploaded_df
    file = request.files.get('file')
    if file is None:
        return "No file uploaded.", 400
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    file.seek(0)
    uploaded_df = pd.read_csv(file, encoding=encoding)
    return uploaded_df.to_html()

@app.route('/remove_special_chars')
def remove_special_chars():
    global uploaded_df

    def clean_special_chars(val):
        if isinstance(val, str):
            return re.sub(r"[^A-Za-z0-9\s\-',&]", '', val)
        return val

    uploaded_df = uploaded_df.applymap(clean_special_chars)
    return uploaded_df.to_html() + "<p><b>Removed special characters (except - ' ,) from text columns.</b></p>"

@app.route('/load_sample')
def load_sample():
    global uploaded_df
    uploaded_df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'bob', 'ALICE', 'Carol', 'Dave', 'dave', 'Eve', 'eve', 'Frank',
             'Grace', 'grace', 'Heidi', 'heidi', 'Ivan', 'IVAN', 'Judy', 'judy', 'Mallory', None],
    'Age': [25, 30, 30, 25, None, 45, 45, 35, None, 40,
            29, 29, None, 50, 55, 55, None, 40, 33, 28],
    'Gender': ['F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M',
               'F', 'F', 'F', None, 'M', 'M', 'F', 'F', None, 'M']
})
    return uploaded_df.to_html()

@app.route('/remove_duplicates_by_column')
def remove_duplicates_by_column():
    global uploaded_df
    column = request.args.get('column')

    if not column:
        return "Error: Column parameter is required.", 400

    if column not in uploaded_df.columns:
        return f"Error: Column '{column}' does not exist in the data.", 400

    before = len(uploaded_df)
    uploaded_df = uploaded_df.drop_duplicates(subset=[column])
    after = len(uploaded_df)
    affected = before - after
    return uploaded_df.to_html() + f"<p><b>Removed {affected} duplicate rows based on column '{column}'.</b></p>"

@app.route('/standardize_text')
def standardize_text():
    global uploaded_df
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
    before = len(uploaded_df)
    uploaded_df = uploaded_df.dropna()
    after = len(uploaded_df)
    affected = before - after
    return uploaded_df.to_html() + f"<p><b>Dropped {affected} rows with missing values.</b></p>"

@app.route('/fill_missing')
def fill_missing():
    global uploaded_df
    before_missing = uploaded_df.isnull().sum().sum()
    uploaded_df = uploaded_df.fillna(0)
    after_missing = uploaded_df.isnull().sum().sum()
    filled = before_missing - after_missing
    return uploaded_df.to_html() + f"<p><b>Filled {filled} missing cells with 0.</b></p>"


@app.route('/sort_values')
def sort_values():
    global uploaded_df
    column = request.args.get('column')

    if column is None or column not in uploaded_df.columns.astype(str).tolist():
        return f"Invalid column name: {html.escape(str(column))}", 400

    uploaded_df.sort_values(by=column, inplace=True)
    return uploaded_df.to_html() + f"<p><b>Sorted by '{html.escape(str(column))}'.</b></p>"

@app.route('/sort_interface')
def sort_interface():
    global uploaded_df

    if uploaded_df.empty:
        return "No data loaded.", 400

    # Ensure all column names are strings
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

    try:
        uploaded_df[column] = uploaded_df[column].astype(dtype)
    except Exception as e:
        return f"Conversion failed: {e}", 400

    return uploaded_df.to_html() + f"<p><b>Changed '{column}' to {dtype}.</b></p>"


@app.route('/replace_values')
def replace_values():
    global uploaded_df, history_stack
    column = request.args.get('column')
    to_replace = request.args.get('replace_from')
    new_value = request.args.get('to')

    if column not in uploaded_df.columns:
        return f"Invalid column: {column}", 400

    # Save current state before replacing
    history_stack.append(uploaded_df.copy())

    uploaded_df[column] = uploaded_df[column].astype(str).str.replace(to_replace, new_value, regex=False)

    return uploaded_df.to_html() + f"<p><b>Replaced '{to_replace}' with '{new_value}' in '{column}'.</b></p>"



@app.route('/rename_column')
def rename_column():
    global uploaded_df
    old_name = request.args.get('old')
    new_name = request.args.get('new')

    if old_name not in uploaded_df.columns:
        return f"Column '{old_name}' not found.", 400

    uploaded_df.rename(columns={old_name: new_name}, inplace=True)
    return uploaded_df.to_html() + f"<p><b>Renamed '{old_name}' to '{new_name}'.</b></p>"


@app.route('/undo_replace')
def undo_replace():
    global uploaded_df, history_stack
    if history_stack:
        uploaded_df = history_stack.pop()
        return uploaded_df.to_html() + "<p><b>Undo successful.</b></p>"
    else:
        return "Nothing to undo.", 400


@app.route('/drop_column')
def drop_column():
    global uploaded_df
    column = request.args.get('column')

    if column not in uploaded_df.columns:
        return f"Column '{column}' not found.", 400

    uploaded_df.drop(columns=[column], inplace=True)
    return uploaded_df.to_html() + f"<p><b>Dropped column '{column}'.</b></p>"


@app.route('/data_profile')
def data_profile():
    global uploaded_df

    if uploaded_df.empty:
        return "No data loaded.", 400

    profile = {}

    # Total records
    profile['Total Records'] = len(uploaded_df)

    # Dynamically check for potential fields
    for col in uploaded_df.columns:
        col_lower = col.lower()

        # Missing values
        missing_count = uploaded_df[col].isnull().sum()
        if missing_count > 0:
            profile[f"Missing values in '{col}'"] = missing_count

        # Phone number validation
        if 'phone' in col_lower:
            pattern = r'^\+?\d{10,15}$'
            invalid_phones = uploaded_df[col].apply(
                lambda x: not bool(re.fullmatch(pattern, str(x))) if pd.notnull(x) else False
            ).sum()
            profile[f"Invalid Phone Numbers in '{col}'"] = invalid_phones

        # Email format validation
        if 'email' in col_lower:
            pattern = r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
            invalid_emails = uploaded_df[col].apply(
                lambda x: not bool(re.fullmatch(pattern, str(x))) if pd.notnull(x) else False
            ).sum()
            profile[f"Invalid Emails in '{col}'"] = invalid_emails

        # Country code format check
        if 'country' in col_lower:
            nonstandard = uploaded_df[col].apply(
                lambda x: not bool(re.fullmatch(r'^[A-Za-z]{2,3}$', str(x))) if pd.notnull(x) else False
            ).sum()
            profile[f"Non-standard Country Codes in '{col}'"] = nonstandard

        # ID field (check for missing)
        if 'id' in col_lower:
            missing_ids = uploaded_df[col].isnull().sum()
            profile[f"Missing IDs in '{col}'"] = missing_ids

    # Duplicates (on full rows)
    profile['Duplicate Rows'] = uploaded_df.duplicated().sum()

    # Format HTML table
    html_output = "<h3>Input Data Profile (Pre-Cleaning)</h3><table border='1'><tr><th>Metric</th><th>Value</th></tr>"
    for key, value in profile.items():
        html_output += f"<tr><td>{key}</td><td>{value}</td></tr>"
    html_output += "</table>"

    return html_output


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host="127.0.0.1", port=5000, debug=True)
