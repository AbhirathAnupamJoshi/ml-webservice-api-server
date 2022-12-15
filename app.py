# Load the libraries
from fastapi import FastAPI, HTTPException
from joblib import load

from sentiment_engine import sentiment_engine

# Initialize an instance of FastAPI
app = FastAPI()

# Load the model
spam_clf = load(open('./models/spam_detector_model.pkl','rb'))

# Load vectorizer
vectorizer = load(open('./vectors/vectorizer.pickle', 'rb'))

@app.get("/")
def root():
    return {"message": "Welcome to ML Web Service API!"}

# Define the default route
@app.get("/predict_sentiment")
def root():
    return {"message": "Welcome to the Sentiment Analysis Model!. Created by Abhirath Anupam Joshi - abhirathjoshi@gmail.com"}

# Define the route to the sentiment predictor
@app.post("/predict_sentiment")
def predict_sentiment(text_message):

    if(not(text_message)):
        raise HTTPException(status_code=400, detail = "Please Provide a valid text message")

    output = sentiment_engine(text_message)

    return {
            "text_message": text_message,
            "Negative Polarity": output[0],
            "Neutral Polarity": output[1],
            "Positive Polarity": output[2],
            "Compound Polarity": output[3],
            "Overall Sentiment": output[4]
           }

# Define the default route
@app.get("/predict_spam")
def root():
    return {"message": "Welcome to the Spam Detection Model!. Created by Abhirath Anupam Joshi - abhirathjoshi@gmail.com"}

# Define the route to the sentiment predictor
@app.post("/predict_spam")
def predict_spam(text_message):

    polarity = ""

    if(not(text_message)):
        raise HTTPException(status_code=400, detail = "Please Provide a valid text message")

    prediction = spam_clf.predict(vectorizer.transform([text_message]))

    if(prediction[0] == 0):
        polarity = "Ham"

    elif(prediction[0] == 1):
        polarity = "Spam"

    return {
            "text_message": text_message,
            "polarity": polarity
           }

if __name__ == '__app__':
    app.run()
