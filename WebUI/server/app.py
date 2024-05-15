from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gensim.utils import simple_preprocess
import gensim
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from gensim.matutils import corpus2csc
from pydantic import BaseModel
from utils import clean
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_methods=["GET", "POST"]
)

# Load the LDA model
lda_model = gensim.models.LdaModel.load("C:\\Users\\RaahimSiddiqi\\Desktop\\Code\\VSC\\FYP\\Models\\lda-combinedv2-50topics-standardfilter-10pass-30iters-0.595cv-20240511T110055Z-001\\lda-combinedv2-50topics-standardfilter-10pass-30iters-0.595cv\\model")
dictionary_path = datapath("C:\\Users\\RaahimSiddiqi\\Desktop\\Code\\VSC\\FYP\\Models\\lda-combinedv2-50topics-standardfilter-10pass-30iters-0.595cv-20240511T110055Z-001\\lda-combinedv2-50topics-standardfilter-10pass-30iters-0.595cv\\model.id2word")
dictionary = Dictionary.load(dictionary_path)

# Load the Random Forest model
rf_model = joblib.load("C:\\Users\\RaahimSiddiqi\\Desktop\\Code\\VSC\\FYP\\Models\\lda-combinedv2-50topics-standardfilter-10pass-30iters-0.595cv-20240511T110055Z-001\\lda-combinedv2-50topics-standardfilter-10pass-30iters-0.595cv\\random_forest_classifier.joblib")


class Document(BaseModel):
    text: str

@app.post("/predict")
async def predict(doc: Document):
    cleaned_text = clean(doc.text)
    lda_sparse_vector = lda_model[dictionary.doc2bow(simple_preprocess(cleaned_text))]

    probs = corpus2csc([lda_sparse_vector], num_terms=50).T.toarray()
    topic_columns = [f"topic {i+1}" for i in range(50)]
    df = pd.DataFrame(probs, columns=topic_columns)

    label = rf_model.predict(df)
    if label[0] == 0:
        label = "non-islamic"
    elif label[0] == 1:
        label = "islamic"
    return {"label": label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
