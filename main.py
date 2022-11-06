from speech_recognition import Recognizer, AudioFile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
# nltk.download("vader_lexicon")


# Save audio speech in text
recognizer = Recognizer()

with AudioFile("chile.wav") as audio_file:
    audio = recognizer.record(audio_file)

# Using Google speech recognition API
text = recognizer.recognize_google(audio)

# Sentiment Analysis of text
analyzer = SentimentIntensityAnalyzer()

# Shows scores for "neg", "neu", "pos", "compound" - the sum of the first 3 coefficients is always equal to 1
# The closer the compound is to 1, the more positive it is
scores = analyzer.polarity_scores(text)
print(scores)

if scores["compound"] > 0:
    print("Positive text")
else:
    print("Negative text")

