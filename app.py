import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from analyzer import RosterAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the logic engine
engine = RosterAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    error = None

    if request.method == 'POST':
        date = request.form['date']
        shift = request.form['shift']
        file = request.files.get('file')

        if not file or file.filename == "":
            error = "Please upload a valid Excel file."
        else:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            try:
                results = engine.get_shift_results(path, date, shift)
            except Exception as e:
                error = f"Processing Error: {str(e)}"
            finally:
                # Defensive Cleanup: Try to delete, but don't crash if Windows locks it
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as cleanup_error:
                    print(f"Warning: Could not delete temp file {path}: {cleanup_error}")

    return render_template('index.html', results=results, error=error)

if __name__ == '__main__':
    app.run(debug=True)