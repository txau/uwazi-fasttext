# export FLASK_APP=ftapi.py
# flask run --port 4000

from flask import Flask, request
import pandas as pd
import json
import subprocess
import os, fcntl
import signal
import random

app = Flask(__name__)

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

def processOneDoc(model, id, text):

    sentences = split_into_sentences(text, 50)
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

def randomize_trianing_set():
    with open('training.txt','r') as source:
        data = [ (random.random(), line) for line in source ]
        data.sort()
    with open('random_training.txt','w') as target:
        for _, line in data:
            target.write( line )

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

    f = open("training.txt", "a")

    try:
        f.write("__label__%s " % label)
        f.write("%s\r\n" % sentence)
    except:
        print 'WARNING: was not successful'
        print data
        print '\n\n'
    f.close()

    return "{}"

@app.route('/classification/retrain', methods=['POST'])
def retrain_route():
    randomize_trianing_set()
    os.system("./fastText/fasttext supervised -pretrainedVectors ./wiki-news-300d-100k-subword.vec -dim 300 -output model -input random_training.txt")
    return "{}"

@app.route('/classification/predictOneModel', methods=['POST'])
def predict_one_model():
    data = json.loads(request.data)
    docs = pd.read_json(json.dumps(data['docs']), encoding='utf-8')

    model = subprocess.Popen(["./fastText/fasttext", "predict-prob", "./model.bin", "-"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    fcntl.fcntl(model.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

    evidences = []
    for index, item in enumerate(docs['_id']):
        evidences += processOneDoc(model, docs['_id'][index], docs['text'][index])

    model.terminate()

    return json.dumps(evidences)

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
