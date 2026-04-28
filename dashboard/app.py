from flask import Flask, render_template_string
import requests

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SLA ML Serving Dashboard</title>
    <style>
        body { font-family: Arial; background: #f4f6f8; padding: 40px; }
        .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px #ccc; max-width: 700px; margin: auto; }
        h1 { color: #222; }
        button { padding: 12px 20px; background: #2563eb; color: white; border: none; border-radius: 8px; cursor: pointer; }
        pre { background: #111; color: #0f0; padding: 20px; border-radius: 8px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="card">
        <h1>SLA-Aware ML Serving Platform</h1>
        <p>This dashboard sends requests to the SLA-aware router.</p>
        <p><b>High Accuracy Model:</b> slower, confidence 0.95</p>
        <p><b>Fast Optimized Model:</b> faster, confidence 0.86</p>

        <form method="POST" action="/predict">
            <button type="submit">Send Prediction Request</button>
        </form>

        {% if result %}
        <h2>Prediction Result</h2>
        <pre>{{ result }}</pre>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML, result=None)

@app.route("/predict", methods=["POST"])
def predict():
    response = requests.post(
        "http://localhost:8000/predict",
        json={"input": "test image"}
    )
    return render_template_string(HTML, result=response.text)

if __name__ == "__main__":
    app.run(port=5000, debug=True)