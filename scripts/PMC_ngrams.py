from datautils.corpusstats import NgramCounter
from nltk.parse.corenlp import CoreNLPParser
from datautils import datapaths
import glob
import requests
import timeit

# NOTE: make sure to open a server for CoreNLP API from a terminal before
# running this script; else, it will crash at the tokenization step
# cf. https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK

TOKENIZER = CoreNLPParser()


def doc_generator(paths):
    for doc in paths:
        try:
            with open(doc) as f:
                text = f.read()
            tokens = list(TOKENIZER.tokenize(text))
            yield tokens
        except requests.exceptions.HTTPError as e:
            print('HTTPError for:', doc)
            continue


# prepare data generator
doc_paths = glob.glob(datapaths.PATH_TO_PMC + '/PMC002XXXXXX.txt/**/*.txt',
                      recursive=True)
docs_to_tokens = doc_generator(doc_paths[:500])

# prepare processors
counter = NgramCounter.from_token_lists(docs_to_tokens, n_docs=500)





