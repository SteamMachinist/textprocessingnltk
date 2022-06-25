import os
import re
import uuid
from collections import defaultdict

import nltk
import fitz
import pymorphy2

from flask import Flask, render_template, request, redirect, url_for
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

UPLOAD_FOLDER = "texts"
ALLOWED_EXTENSIONS = {'.pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024

nltk.download('punkt')
nltk.download("stopwords")


stops = set(stopwords.words("russian"))
stemmer_rus = SnowballStemmer("russian")
morph = pymorphy2.MorphAnalyzer()


def get_all_text(folder):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    file_list = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], folder))
    all_text = ""

    for file in file_list:
        with fitz.open(os.path.join(folder_path, file)) as pdf:
            for page in pdf:
                all_text += page.get_text()
    return all_text


def tokenize(all_text: str) -> list:
    return [word.lower() for word in nltk.word_tokenize(all_text) if word.isalpha() and word.lower() not in stops]


def filter_tokens(tokens: list) -> list:
    return [word.lower() for word in tokens if word.isalpha() and word.lower() not in stops]


def processed_tokens_to_sorted_dictionary(processed_tokens: list) -> dict:
    token_dict = defaultdict(int)
    for token in processed_tokens:
        token_dict[token] += 1
    return dict(sorted(token_dict.items(), key=lambda item: item[1], reverse=True))


def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))


@app.route('/result/s/<folder>')
def get_stemming_result(folder: str):
    tokenized = tokenize(get_all_text(folder))
    stemmed_tokens = []
    for token in tokenized:
        if has_cyrillic(token):
            stemmed_tokens.append(stemmer_rus.stem(token))
        else:
            stemmed_tokens.append(token)
    return render_template('result.html',
                           process_type="стемминга",
                           files_list=str(os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], folder))),
                           tokens=processed_tokens_to_sorted_dictionary(stemmed_tokens))


@app.route('/result/l/<folder>')
def get_lemmatization_results(folder: str):
    lemmatized = [morph.parse(token)[0].normal_form for token in tokenize(get_all_text(folder))]
    return render_template('result.html',
                           process_type="лемматизации",
                           files_list=str(os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], folder))),
                           tokens=processed_tokens_to_sorted_dictionary(lemmatized))


def is_valid(filename: str) -> bool:
    return filename != '' and os.path.splitext(filename)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def upload_files_index():
    if request.method == 'GET':
        return render_template('start.html', error_message="")

    if "process_example" in request.form:
        folder = 'example'
    else:
        files = request.files.getlist('files')
        if not all([is_valid(file.filename) for file in files]):
            return render_template('start.html', error_message="Файлы должны иметь формат .pdf")

        folder = uuid.uuid4().hex
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
        os.mkdir(folder_path)

        for file in files:
            file.save(os.path.join(folder_path, file.filename))

    process_type = request.form['process_type']
    if process_type == 'stemming':
        return redirect(url_for('get_stemming_result', folder=folder))
    else:
        return redirect(url_for('get_lemmatization_results', folder=folder))


if __name__ == '__main__':
    app.run()
