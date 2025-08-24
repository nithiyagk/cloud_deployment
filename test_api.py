import requests

# Use your deployed URL if running on Render/Heroku, otherwise use localhost
url = "http://127.0.0.1:5000/predict"   # change if running on cloud

# Example features (make sure number matches your model input)
data = {"features": [5.1, 3.5, 1.4, 0.2]}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Raw Response:", response.text)  # useful for debugging
try:
    print("JSON Response:", response.json())
except Exception:
    print("Response is not valid JSON")
