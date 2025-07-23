from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
import csv
import io
import threading
from llm_cyberbullying_classifier import classify_message

app = Flask(__name__)

# In-memory stats and last CSV results
dashboard_stats = {
    'total_entries': 0,
    'bullying_cases': 0
}
last_csv_results = io.StringIO()
last_csv_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/stats')
def stats():
    return jsonify({
        'total_entries': dashboard_stats['total_entries'],
        'bullying_cases': dashboard_stats['bullying_cases']
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('tweet', '')
    result = classify_message(text)
    label, confidence, explanation, highlights = parse_llm_result(result)
    dashboard_stats['total_entries'] += 1
    is_bullying = label.lower() == 'yes'
    if is_bullying:
        dashboard_stats['bullying_cases'] += 1
    label_out = 'cyberbullying' if is_bullying else 'not_cyberbullying'
    return jsonify({
        'label': label_out,
        'confidence': confidence,
        'explanation': explanation,
        'highlights': highlights,
        'explainable': explanation
    })

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'csv' not in request.files:
        return 'No file uploaded', 400
    file = request.files['csv']
    if not file.filename.endswith('.csv'):
        return 'Invalid file type', 400
    file_stream = io.StringIO(file.stream.read().decode('utf-8'))
    reader = csv.reader(file_stream)
    rows = list(reader)
    header = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []
    results = []
    for row in data_rows:
        text = row[0]
        result = classify_message(text)
        label, confidence, explanation, highlights = parse_llm_result(result)
        is_bullying = label.lower() == 'yes'
        label_out = 'cyberbullying' if is_bullying else 'not_cyberbullying'
        results.append([text, label_out, confidence, explanation, highlights])
        dashboard_stats['total_entries'] += 1
        if is_bullying:
            dashboard_stats['bullying_cases'] += 1
    # Save for download
    with last_csv_lock:
        last_csv_results.seek(0)
        last_csv_results.truncate()
        writer = csv.writer(last_csv_results)
        writer.writerow(['Text', 'Label', 'Confidence', 'Explanation', 'Highlights'])
        writer.writerows(results)
    # Render HTML table
    html = '<table class="table table-striped table-dark"><thead><tr>'
    for col in ['Text', 'Label', 'Confidence', 'Explanation', 'Highlights']:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    for row in results:
        html += '<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>'
    html += '</tbody></table>'
    return html

@app.route('/download')
def download():
    with last_csv_lock:
        last_csv_results.seek(0)
        return send_file(io.BytesIO(last_csv_results.getvalue().encode('utf-8')),
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='cyberbullying_results.csv')

def parse_llm_result(result):
    # Expecting result in format:
    # Cyberbullying: <Yes/No>\nConfidence: <0-100>%\nExplanation: <short explanation>\nHighlights: <comma-separated keywords/phrases>
    label = confidence = explanation = highlights = ''
    for line in result.split('\n'):
        if line.lower().startswith('cyberbullying:'):
            label = line.split(':', 1)[-1].strip()
        elif line.lower().startswith('confidence:'):
            conf = line.split(':', 1)[-1].strip().replace('%', '')
            try:
                confidence = float(conf)
            except:
                confidence = 0.0
        elif line.lower().startswith('explanation:'):
            explanation = line.split(':', 1)[-1].strip()
        elif line.lower().startswith('highlights:'):
            highlights = line.split(':', 1)[-1].strip()
    return label, confidence, explanation, highlights

if __name__ == '__main__':
    app.run(debug=True) 