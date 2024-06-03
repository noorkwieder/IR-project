import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import datefinder
import spacy
import datetime

nlp = spacy.load('en_core_web_sm')

# Define the abbreviation dictionary
abbreviations = {
    'dr.': 'doctor',
    'jr.': 'junior',
    'mr.': 'mister',
    'mrs.': 'missus',
    'ms.': 'miss',
    'gov.': 'governor',
    'sen.': 'senator',
    'rep.': 'representative',
    'dept.': 'department',
    'univ.': 'university',
    'col.': 'colonel',
    'sgt.': 'sergeant',
    'capt.': 'captain',
    'prof.': 'professor',
    'eng.': 'engineer',
    'pharm.': 'pharmacist',
    'acct.': 'accountant',
    'admin.': 'administrator',
    'asst.': 'assistant',
    'attys.': 'attorneys',
    'co.': 'company',
    'corp.': 'corporation',
    'doc.': 'document',
    'gen.': 'general',
    'inc.': 'incorporated',
    'intl.': 'international',
    'mgr.': 'manager',
    'msg.': 'message',
    'no.': 'number',
    'pg.': 'page',
    'pt.': 'part',
    'prod.': 'product',
    'proj.': 'project',
    'sec.': 'secretary',
    'tech.': 'technician',
    'temp.': 'temporary',
    'USA': 'United States of America',
    'UK': 'United Kingdom',
    'UAE': 'United Arab Emirates',
    'CAN': 'Canada',
    'AUS': 'Australia',
    'GER': 'Germany',
    'FRA': 'France',
    'ITA': 'Italy',
    'JPN': 'Japan',
    'CHN': 'China',
    'IND': 'India',
    'BRA': 'Brazil',
    'RUS': 'Russia',
    'MEX': 'Mexico',
    'ESP': 'Spain',
    'ARG': 'Argentina',
    'SA': 'Saudi Arabia',
    'EGY': 'Egypt',
    'NGA': 'Nigeria',
    'ZAF': 'South Africa',
    'u': 'you'
    # Add more abbreviations as needed
}

date_formats = [
    "%Y-%m-%d",        # Year-Month-Day (e.g., 2023-05-30)
    "%d-%m-%Y",        # Day-Month-Year (e.g., 30-05-2023)
    "%m-%d-%Y",        # Month-Day-Year (e.g., 05-30-2023)
    "%Y/%m/%d",        # Year/Month/Day (e.g., 2023/05/30)
    "%d/%m/%Y",        # Day/Month/Year (e.g., 30/05/2023)
    "%m/%d/%Y",        # Month/Day/Year (e.g., 05/30/2023)
    "%Y.%m.%d",        # Year.Month.Day (e.g., 2023.05.30)
    "%d.%m.%Y",        # Day.Month.Year (e.g., 30.05.2023)
    "%m.%d.%Y",        # Month.Day.Year (e.g., 05.30.2023)
    "%Y_%m_%d",        # Year_Month_Day (e.g., 2023_05_30)
    "%d_%m_%Y",        # Day_Month_Year (e.g., 30_05_2023)
    "%m_%d_%Y",        # Month_Day_Year (e.g., 05_30_2023)
    "%d-%b-%Y",        # Day-Month Abbreviation-Year (e.g., 1-Jan-2020)
    "%b-%d-%Y",        # Month Abbreviation-Day-Year (e.g., Jan-1-2020)
    "%d-%B-%Y",        # Day-Month Full Name-Year (e.g., 1-January-2020)
    "%B-%d-%Y",        # Month Full Name-Day-Year (e.g., January-1-2020)
    "%d.%b.%Y",        # Day.Month.Abbreviation.Year (e.g., 1.Jan.2020)
    "%b.%d.%Y",        # Abbreviation.Day.Month.Year (e.g., Jan.1.2020)
    "%d.%B.%Y",        # Day.Month Full Name.Year (e.g., 1.January.2020)
    "%B.%d.%Y",        # Month Full Name.Day.Year (e.g., January.1.2020)
    "%d/%b/%Y",        # Day/Month Abbreviation/Year (e.g., 1/Jan/2020)
    "%b/%d/%Y",        # Month Abbreviation/Day/Year (e.g., Jan/1/2020)
    "%d/%B/%Y",        # Day/Month Full Name/Year (e.g., 1/January/2020)
    "%B/%d/%Y",        # Month Full Name/Day/Year (e.g., January/1/2020)
    "%d_%b_%Y",        # Day_Month Abbreviation_Year (e.g., 1_Jan_2020)
    "%b_%d_%Y",        # Month Abbreviation_Day_Year (e.g., Jan_1_2020)
    "%d_%B_%Y",        # Day_Month Full Name_Year (e.g., 1_January_2020)
    "%B_%d_%Y",        # Month Full Name_Day_Year (e.g., January_1_2020)
]

def tokenize(text):
    return word_tokenize(text)

def lowercase(tokens):
    return [token.lower() for token in tokens]

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def expand_abbreviations(tokens):
    return [abbreviations.get(token.lower(), abbreviations.get(token.upper(), token.lower())) for token in tokens]

def convert_dates(tokens):
    converted_tokens = []
    for token in tokens:
        for date_format in date_formats:
            matches = datefinder.find_dates(token, source=True)
            for match in matches:
                converted_token = match[0].strftime("%Y-%m-%d %H:%M:%S")
                token = token.replace(match[1], converted_token)
        converted_tokens.append(token)
    return converted_tokens

def lemmatize(tokens):
    lemmatized_tokens = []
    for token in tokens:
        try:
            datetime.datetime.strptime(token, '%Y-%m-%d %H:%M:%S')
            lemmatized_token = token
        except ValueError:
            doc = nlp(token)
            lemmatized_token = doc[0].lemma_
        lemmatized_tokens.append(lemmatized_token)
    return lemmatized_tokens

def stem(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token.strip()) for token in tokens]

def remove_special_chars(tokens):
    return [re.sub(r'[^A-Za-z0-9]+', '', token) for token in tokens]

def remove_empty(tokens):
    return [token for token in tokens if token]

def spell_correction(tokens):
    corrected_tokens = []
    for token in tokens:
        blob = TextBlob(token)
        corrected_token = str(blob.correct())
        corrected_tokens.append(corrected_token)
    return corrected_tokens


import csv


def process_documents(file_path, output_file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        documents = file.readlines()  # Read all lines in the file

    with open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(['ID', 'Content'])  # Write the header row

        for idx, document_text in enumerate(documents, start=1):
            # Optionally, you can add a check to skip empty lines or handle edge cases
            if document_text.strip():  # Skip empty lines
                # Extract the ID from the beginning of the document text
                document_id, document_content = extract_id_and_content(document_text)

                # Process the document content (tokenization or any other processing)
                tokens = process_document_text(document_content)

                # Write the ID and processed content to the CSV file
                csv_writer.writerow([document_id, ' '.join(tokens)])


def extract_id_and_content(document_text):
    # Split the document text to separate ID and content
    parts = document_text.split(maxsplit=1)
    if len(parts) > 1:
        document_id = parts[0]
        document_content = parts[1].strip()  # Remove leading/trailing whitespaces
        return document_id, document_content


def process_document_text(document_text):
    # Implement your document processing function here
    # Example processing steps:
    tokens = tokenize(document_text)
    # print('step1')
    tokens = lowercase(tokens)
    # print('step2')
    tokens = expand_abbreviations(tokens)
    # print('step3')
    tokens = convert_dates(tokens)
    # print('step4')
    tokens = lemmatize(tokens)
    # print('step5')
    tokens = stem(tokens)
    # print('step6')
    tokens = remove_stopwords(tokens)
    # print('step7')
    tokens = remove_special_chars(tokens)
    # print('step8')
    tokens = remove_empty(tokens)
    # print('step9')
    tokens = spell_correction(tokens)
    # print('step10')

    return tokens

# Set the paths to your dataset and the output file
dataset_path = "C:\\Users\\HP\\.ir_datasets\\antique\\test.tsv"
output_file_path = "C:\\Users\\HP\\.ir_datasets\\antique\\processed_output_test.csv"

process_documents(dataset_path, output_file_path)