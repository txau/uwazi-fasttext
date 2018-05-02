# export TFHUB_CACHE_DIR=./tfhub_modules
# export FLASK_APP=ft_routes.py
# flask run --port 4000

from flask import Flask, request
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import subprocess
import os, fcntl
import signal
import random

app = Flask(__name__)
vocabSize = 40000
THRESHOLD = 0.67
SENTENCE_LENGTH = 30

TRAINING_FILE = 'training.csv'
RANDOMIZED_TRAINING_FILE = 'random_training.txt'

sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

def prepro(sentence):
    result = sentence.replace(",", " , ")
    result = result.replace(":", " : ")
    result = result.replace(";", " ; ")
    return result;

def split_into_sentences(text, size):
    words = text.split()

    arrs = []
    while len(words) > size:
        piece = words[:size]
        arrs.append(piece)
        words = words[size:]
    arrs.append(words)

    sentences = []
    for one_sentence in arrs:
        one_sentence = ' '.join(one_sentence)
        sentences.append(one_sentence)

    return sentences

def display_sentence_result(model, sentence):
    model.stdin.write(sentence.encode('utf-8') + "\n")
    while True:
        try:
            return model.stdout.read()
        except Exception:
            pass


def get_similar_sentences(similarity, evidences, sentences, doc_id):
    similar_sentences = pd.DataFrame(columns=['probability'])
    for ind, sentence in enumerate(sentences):
        for pos,sim in enumerate(similarity[:,ind]):
            if sim>=THRESHOLD:
                similar_sentences = similar_sentences.append({'evidence':sentence, 'probability':sim, 'label':1, 'document':doc_id, 'property':evidences.loc[pos]['property'], 'value':evidences.loc[pos]['value']}, ignore_index=True)
    return similar_sentences


def processOneDoc(model, id, text):

    sentences = split_into_sentences(text, SENTENCE_LENGTH)
    evidences = []

    for one_sentence in sentences:
        one_result = {}
        one_result['evidence'] = one_sentence
        result = display_sentence_result(model, prepro(one_sentence))
        parts = result.split()
        one_result['probability'] = parts[1]
        info = parts[0].split("_")
        one_result['property'] = info[4]
        one_result['value'] = info[5]
        one_result['label'] = int(info[6] == 'True')
        one_result['document'] = id
        if one_result['label'] == 1:
            evidences.append(one_result)

    return evidences

def randomize_training_set():
    training_data = pd.read_csv(TRAINING_FILE, encoding='utf8')
    training_data['isEvidence'] = training_data['isEvidence'].apply(str)
    training_data['label'] = training_data[['property', 'value', 'isEvidence']].apply(lambda x: '_'.join(x), axis=1)

    randomized_training = training_data.sample(frac=1)
    randomized_training[['label', 'sentence']].to_csv(RANDOMIZED_TRAINING_FILE, sep=' ', index=False, header=False, encoding='utf8')


@app.route('/predict-prob', methods=['POST'])
def predict_prob_route():
    data = json.loads(request.data)
    p.stdin.write(data['sentence'] + "\n")
    while True:
        try:
            return p.stdout.read()
        except Exception:
            pass

@app.route('/classification/train', methods=['POST'])
def train_route():
    data = json.loads(request.data)
    label = data['property'] + "_" + data['value'] + "_" + str(data['isEvidence']);
    sentence = data['evidence']['text'].encode('utf-8');
    sentence = prepro(sentence)

    if data['isEvidence']:
        df = pd.DataFrame({'sentence': sentence, 'property':data['property'], 'value':data['value'], 'isEvidence':data['isEvidence']}, index=[0])
        if os.path.exists(TRAINING_FILE):
            df.to_csv(TRAINING_FILE, mode='a', header=False, index=False, encoding='utf8')
        else:
            df.to_csv(TRAINING_FILE, mode='a', index=False, encoding='utf8')

    return "{}"


@app.route('/classification/retrain', methods=['POST'])
def retrain_route():
    randomize_training_set()
    os.system("./fastText/fasttext supervised -pretrainedVectors ./wiki-news-300d-{0}-subword.vec -dim 300 -output model -input {1}".format(vocabSize, RANDOMIZED_TRAINING_FILE))
    return "{}"


@app.route('/classification/predictOneModel', methods=['POST'])
def predict_one_model():
    evidences = pd.read_csv(TRAINING_FILE, encoding='utf8')

    data = json.loads(request.data)
    docs = pd.read_json(json.dumps(data['docs']), encoding='utf-8')

    if len(evidences) >= 20:
        model = subprocess.Popen(["./fastText/fasttext", "predict-prob", "./model.bin", "-"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        fcntl.fcntl(model.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

        suggestions = []
        for index, item in enumerate(docs['_id']):
            suggestions += processOneDoc(model, docs['_id'][index], docs['text'][index])

        model.terminate()
        return json.dumps(suggestions)

    else:
        model_evidences = evidences[(evidences['property']==data['property']) & (evidences['value']==data['value'])]
        model_evidences = model_evidences.reset_index()

        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])

            evidence_embedding = session.run(sentenceEncoder(model_evidences['sentence'].tolist()))
            suggestions = pd.DataFrame(columns=['probability'])
            for doc in docs.iterrows():
                sentences = split_into_sentences(doc[1].text, SENTENCE_LENGTH)
                sentence_embedding = session.run(sentenceEncoder(sentences))

                similarity = np.matmul(evidence_embedding, np.transpose(sentence_embedding))
                suggestions = suggestions.append(get_similar_sentences(similarity, evidences, sentences, doc[1]['_id']))
                suggestions.sort_values(by=['probability'], ascending=False, inplace=True)
                suggestions.drop_duplicates(inplace=True)
        return suggestions.to_json(orient='records')


@app.route('/classification/predict', methods=['POST'])
def predict_route():
    data = json.loads(request.data)
    evidencesData = data['properties']
    doc = pd.read_json('[' + json.dumps(data['doc']) + ']', encoding='utf8').loc[0];

    model = subprocess.Popen(["./fastText/fasttext", "predict-prob", "./model.bin", "-"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    fcntl.fcntl(model.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

    evidences = processOneDoc(model, evidencesData[0]['document'], doc['text'])

    model.terminate()

    return json.dumps(evidences)
