# export FLASK_APP=ft_routes.py
# flask run --port 4000

from flask import Flask, request
import pandas as pd
import json
import subprocess
import os, fcntl
import signal
import random
import sys

reload(sys)
sys.setdefaultencoding('utf8')

app = Flask(__name__)
sentence_dictionary = ''

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
        one_result['predictedLabel'] = int(info[6] == 'True')
        one_result['document'] = id
        if one_result['predictedLabel'] == 1 and not check_dupp2(one_result):
            evidences.append(one_result)

    return evidences

def check_dupp(one_sentence):
    initialise_sentence_dictionary()
    global sentence_dictionary

    sentence = one_sentence['evidence'].encode('utf-8').replace('"', '\\"')

    query = sentence_dictionary.query(   'document == "' + one_sentence['document']
                        + '" and property == "' + one_sentence['property'] + '"'
                        + ' and value == "' + one_sentence['value'] + '"'
                        + ' and isEvidence == "' + str(one_sentence['isEvidence']) + '"'
                        + ' and sentence == "' + sentence + '"')

    if not query.empty:
        return True

    return False

def check_dupp2(one_sentence):
    initialise_sentence_dictionary()
    global sentence_dictionary

    sentence = one_sentence['evidence'].encode('utf-8').replace('"', '\\"')

    query = sentence_dictionary.query(   'document == "' + one_sentence['document']
                        + '" and property == "' + one_sentence['property'] + '"'
                        + ' and value == "' + one_sentence['value'] + '"'
                        + ' and isEvidence == "' + str(bool(one_sentence['predictedLabel'])) + '"'
                        + ' and sentence == "' + sentence + '"')

    if not query.empty:
        return True

    return False

def word_heatmap(model, evidence):
    words = evidence['evidence'].split()

    heatmap = {'probability': evidence['probability'], 'top_words': []}

    position = 0
    while position < len(words):
        words_left = words[:position]
        da_word = words[position:position+1]
        words_right = words[position +1:]
        probe = ' '.join(words_left) + ' ' + ' '.join(words_right)
        result = display_sentence_result(model, prepro(probe))
        parts = result.split()
        position += 1
        heatmap['top_words'].append({'word': da_word, 'probability': parts[1]})

    sorted_obj = dict(heatmap)
    sorted_obj['top_words'] = sorted(heatmap['top_words'], key=lambda x : x['probability'])

    return sorted_obj

def initialise_sentence_dictionary():
    global sentence_dictionary

    if not isinstance(sentence_dictionary, pd.DataFrame):
        try:
            with open("memory.csv") as file:
                file.close()
                sentence_dictionary = pd.read_csv("memory.csv")
        except IOError:
            sentence_dictionary = pd.DataFrame()

def raw_dictionary_entry():
    return pd.DataFrame([{'document':'', 'property':'', 'value':'', 'isEvidence':'', 'sentence':''}])

@app.route('/classification/train', methods=['POST'])
def train_route():
    data = json.loads(request.data)
    data['evidence'] =  data['evidence']['text'].encode('utf-8')

    initialise_sentence_dictionary()
    global sentence_dictionary

    if not sentence_dictionary.empty and check_dupp(data):
        print "Dupp sentence, not adding"
        return "{}"

    entry = raw_dictionary_entry()

    entry.loc[0, 'document'] =  data['document']
    entry.loc[0, 'property'] =  data['property']
    entry.loc[0, 'value'] =  data['value']
    entry.loc[0, 'isEvidence'] = str(data['isEvidence'])
    entry.loc[0, 'sentence'] =  data['evidence']

    sentence_dictionary = sentence_dictionary.append(entry, ignore_index = True)
    sentence_dictionary.to_csv("memory.csv")

    return "{}"

@app.route('/classification/retrain', methods=['POST'])
def retrain_route():
    initialise_sentence_dictionary()
    global sentence_dictionary
    sentence_dictionary.sample(frac=1)

    f = open("training.txt", "w")
    for index, row in sentence_dictionary.iterrows():
        label = "__label__" + row['property'] + "_" + row['value'] + "_" + str(row['isEvidence'])
        sentence = row['sentence'].encode('utf-8');
        sentence = prepro(sentence)
        try:
            f.write(label)
            f.write(" %s\r\n" % sentence)
        except:
            print 'WARNING: was not successful'
            print data
            print '\n\n'

    f.close()

    os.system("./fastText/fasttext supervised -pretrainedVectors ./wiki-news-300d-10k-subword.vec -dim 300 -output model -input training.txt")
    return "{}"

@app.route('/classification/predictOneModel', methods=['POST'])
def predict_one_model():
    data = json.loads(request.data)
    docs = pd.read_json(json.dumps(data['docs']), encoding='utf-8')

    initialise_sentence_dictionary()
    global sentence_dictionary

    model = subprocess.Popen(["./fastText/fasttext", "predict-prob", "./model.bin", "-"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    fcntl.fcntl(model.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

    evidences = []
    for index, item in enumerate(docs['_id']):
        evidences += processOneDoc(model, docs['_id'][index], docs['text'][index])
        if len(evidences) > 100:
            break

    for one_evidence in evidences:
        one_evidence['options'] = word_heatmap(model, one_evidence)

    print evidences

    model.terminate()

    return json.dumps(evidences)

@app.route('/classification/predict', methods=['POST'])
def predict_route():

    return {}
