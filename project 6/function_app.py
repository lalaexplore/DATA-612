import azure.functions as func
import pickle
import pandas as pd
import numpy as np
import json
import os
import tempfile
from azure.storage.blob import BlobClient

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

CONTAINER = "datasets"


# ---------- Download helper ----------
def download_blob_to_file(blob_name, local_path):
    blob = BlobClient.from_connection_string(
        CONN_STR,
        CONTAINER,
        blob_name
    )

    with open(local_path, "wb") as f:
        f.write(blob.download_blob().readall())

# ---------- Load model once per Function instance ----------
tmp_dir = tempfile.gettempdir()

model_path = os.path.join(tmp_dir, "model.pkl")
books_path = os.path.join(tmp_dir, "books.csv")

if not os.path.exists(model_path):
    download_blob_to_file("svd_sgd_model.pkl", model_path)

if not os.path.exists(books_path):
    download_blob_to_file("books_lookup.csv", books_path)

with open(model_path, "rb") as f:
    model = pickle.load(f)

books = pd.read_csv(books_path, dtype={"ISBN": str})


# ---------- HTTP Trigger ----------
@app.route(route="recommend")
def recommend(req: func.HttpRequest) -> func.HttpResponse:

    try:
        user_id_param = req.params.get("user_id")

        if not user_id_param:
            return func.HttpResponse(
                "Missing user_id",
                status_code=400
            )

        user_id = int(user_id_param)

        # Check if the user exists
        if user_id not in model["user_to_idx"]:
            return func.HttpResponse(
                "Unknown user",
                status_code=400
            )

        u = model["user_to_idx"][user_id]

        # Predict ratings
        scores = (
            model["mu"]
            + model["bu"][u]
            + model["bi"]
            + model["Q"] @ model["P"][u]
        )

        # Get top 10 recommendations
        top_idx = np.argsort(scores)[::-1][:10]
        top_books = [model["idx_to_book"][i] for i in top_idx]

        results = []

        for isbn, score in zip(top_books, scores[top_idx]):
            row = books[books["ISBN"] == isbn]

            if not row.empty:
                results.append({
                    "ISBN": isbn,
                    "title": row["Title"].values[0],
                    "author": row["Author"].values[0],
                    "predicted_rating": float(score)
                })

        return func.HttpResponse(
            json.dumps(results),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=500
        )