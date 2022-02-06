from django.apps import AppConfig
from django.conf import settings
import os
import pickle
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification


class ReviewanalyzerConfig(AppConfig):
    name = 'reviewanalyzer'

    path_sentiment = os.path.join(settings.MODELS, 'classifier_BERT')
    path_spam = os.path.join(settings.MODELS, 'spam_classifier.p')

    model_sentiment = TFDistilBertForSequenceClassification.from_pretrained(path_sentiment)
    vectorizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    labels = ['Negative', 'Positive']

    with open(path_spam, 'rb') as p:
        data = pickle.load(p)
    model_spam = data['model']
    tokenizer = data['tokenizer']
    max_len = data['maxlen']



