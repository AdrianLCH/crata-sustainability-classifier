import requests
import pandas as pd

URL = "http://localhost:5000/predict"

def predict_from_text():
    text_description = input("Please enter text to make a prediction: ")
    return [text_description]

def predict_from_csv():
    csv_path = input("Please enter the path of the CSV file: ")
    try:
        df = pd.read_csv(csv_path, nrows=20)
    except FileNotFoundError:
        print("The CSV file was not found at the specified path.")
        return None, None

    if "about" not in df.columns:
        print("The 'about' column was not found in the CSV file.")
        return None, None

    text_descriptions = df["about"].tolist()
    ids = df["id"].tolist() if "id" in df.columns else list(range(len(text_descriptions)))
    return text_descriptions, ids

def predict_batch(texts, ids):
    for text, company_id in zip(texts, ids):
        response = requests.post(URL, json={"text_description": text})
        if response.status_code == 200:
            pred = response.json().get("prediction")
            print(f"Company {company_id}: {'YES sustainable' if pred == 1 else 'NO sustainable'}")
        else:
            print(f"Error for company {company_id}. Status code: {response.status_code}")

def predict_single(text):
    response = requests.post(URL, json={"text_description": text})
    if response.status_code == 200:
        pred = response.json().get("prediction")
        print("My prediction is YES sustainable" if pred == 1 else "My prediction is NO sustainable")
    else:
        print("Error sending request. Status code:", response.status_code)

if __name__ == "__main__":
    option = input("Choose an option:\n1. Enter text manually\n2. Load a CSV file\n")

    if option == "1":
        texts = predict_from_text()
        predict_single(texts[0])
    elif option == "2":
        texts, ids = predict_from_csv()
        if texts:
            predict_batch(texts, ids)
    else:
        print("Invalid option. Please choose 1 or 2.")