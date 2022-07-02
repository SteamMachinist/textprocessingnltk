import os
import uuid
import csv
from collections import defaultdict

import nltk
import fitz
import pymorphy2

from flask import Flask, render_template, request, redirect, url_for, send_file
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


def get_all_pdf_text(folder_path, files_list):
    all_text = ""
    for file in files_list:
        with fitz.open(os.path.join(folder_path, file)) as pdf:
            for page in pdf:
                all_text += page.get_text()
    return all_text


def tokenize(all_text: str) -> list:
    return [word.lower() for word in nltk.word_tokenize(all_text) if word.isalpha() and word.lower() not in stops]


def processed_tokens_to_sorted_dictionary(processed_tokens: list) -> dict:
    token_dict = defaultdict(int)
    for token in processed_tokens:
        token_dict[token] += 1
    return dict(sorted(token_dict.items(), key=lambda item: item[1], reverse=True))


def process(files_list, folder_path: str, folder: str, process_type: str):
    files_list.remove('result.csv') if 'result.csv' in files_list else None
    files_list.remove('processed.txt') if 'processed.txt' in files_list else None

    tokenized = tokenize(get_all_pdf_text(folder_path, files_list))

    if process_type == 'lem':
        processed_tokens = [morph.parse(token)[0].normal_form for token in tokenized]
    elif process_type == 'stem':
        processed_tokens = [stemmer_rus.stem(token) for token in tokenized]
    else:
        processed_tokens = tokenized

    with open(os.path.join(folder_path, 'processed.txt'), mode='w+', encoding="utf-8") as processed:
        processed.write('\n'.join(files_list))
    with open(os.path.join(folder_path, 'result.csv'), mode='w+', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        csv_writer.writerows(processed_tokens_to_sorted_dictionary(processed_tokens).items())

    if folder != 'example':
        for file in files_list:
            os.remove(os.path.join(folder_path, file))


@app.route('/result/<folder>/<process_type>', methods=['GET'])
def get_result(folder: str, process_type: str):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    files_list = os.listdir(folder_path)
    if 'result.csv' not in files_list or folder == 'example':
        process(files_list, folder_path, folder, process_type)

    with open(os.path.join(folder_path, 'processed.txt'), mode='r', encoding="utf-8") as processed:
        files_list = ', '.join(processed.read().split('\n'))
    with open(os.path.join(folder_path, 'result.csv'), mode='r', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        tokens_with_amount = [(row[0], row[1]) for row in csv_reader]

    return render_template('result.html',
                           files_list=files_list,
                           folder=folder,
                           process_type=process_type,
                           tokens_with_amount=tokens_with_amount)


@app.route('/result/<folder>/<process_type>/download', methods=['GET'])
def download_result(folder: str, process_type: str):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], folder, 'result.csv'), as_attachment=True)


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

    return redirect(url_for('get_result', folder=folder, process_type=request.form['process_type']))


if __name__ == '__main__':
    app.run()
