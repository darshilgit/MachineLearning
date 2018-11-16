import re
import os
from os.path import join
from os import listdir
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import math

nltk.download('stopwords')
nltk.download('punkt')

data_path = './20_newsgroups/'
stop_words = set(stopwords.words('english'))


training_data_size = 500
punctuations_and_special_characters = '[!|@|#|$|%|^|&|*|(|)|{|}|;|:|[|,|.|/|<|>|?|\|||`|~|-|=|_|+]'

folders = [folder for folder in listdir(data_path)]
files = []
for folder in folders:
    folder_path = join(data_path, folder)
    files.append([file for file in listdir(folder_path)])

folder_to_word_to_count = defaultdict(dict)
word_count_in_a_folder = defaultdict(dict)
prior = defaultdict(dict)
folder_to_doc_count = defaultdict(dict)


def do_preprocessing(path):
    file = open(path, 'rb')
    text = file.read()
    text = text.decode("ISO-8859-1")
    text = re.sub(punctuations_and_special_characters, '', text)
    word_tokens = word_tokenize(text)
    return word_tokens


def do_training():
    folder_number = 0
    doc_count = 0
    for file in files:
        if file:
            folder_name = folders[folder_number]
            folder_to_word_to_count[folder_name] = {}
            word_count_in_a_folder[folder_name] = 0
            folder_to_doc_count[folder_name] = 0
            file_number = 1
            for file_instance in file:
                if file_number > training_data_size:
                    break
                doc_count += 1
                folder_to_doc_count[folder_name] += 1
                path = join(data_path, join(folders[folder_number], file_instance))
                word_tokens = do_preprocessing(path)

                for word in word_tokens:
                    word = word.lower()
                    if word not in stop_words:
                        word_count_in_a_folder[folder_name] +=1
                        if word in folder_to_word_to_count[folder_name]:
                            folder_to_word_to_count[folder_name][word] +=1
                        else:
                            folder_to_word_to_count[folder_name][word] = 1
                file_number += 1
            folder_number += 1
    return float(doc_count)


def do_testing():
    no_of_classes = len(folders)
    folder_number = 0
    match_count = 0
    folder_to_probability = defaultdict(dict)

    # prior prob  = number of docs in class / total docs
    for folder in folder_to_word_to_count:
        prior[folder] = float(folder_to_doc_count[folder]) / float(doc_count)

    for file in files:
        if file:
            folder_name = folders[folder_number]
            file_number = 0
            for file_instance in file:
                file_number +=1
                if not (file_number < training_data_size):
                    actual_class = os.path.basename(os.path.normpath(folder_name))
                    path = join(data_path, join(folders[folder_number], file_instance))
                    word_tokens = do_preprocessing(path)
                    for folder in folder_to_word_to_count:
                        folder_to_probability[folder] = math.log10(prior[folder])
                        for word in word_tokens:
                            word = word.lower()
                            if word not in stop_words:
                                if word in folder_to_word_to_count[folder]:
                                    probability = float(folder_to_word_to_count[folder][word]) / float(word_count_in_a_folder[folder])
                                else:
                                    probability = float(1) / float(word_count_in_a_folder[folder])
                                folder_to_probability[folder] += math.log10(probability)
                    predicted_class = max(folder_to_probability, key=folder_to_probability.get)
                    if actual_class == predicted_class:
                        match_count += 1
            folder_number += 1
    return float(match_count)


doc_count = do_training()
match_count = do_testing()
accuracy = (match_count/doc_count) * 100
print('Accuracy:', accuracy)
