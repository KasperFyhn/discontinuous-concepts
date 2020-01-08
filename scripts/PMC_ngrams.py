from datautils.corpusstats import NgramCounter, CorpusStats
from nltk.parse.corenlp import CoreNLPParser
from datautils import datapaths
import glob
import requests
import os
from pathlib import Path
import multiprocessing as mp

import timeit

"""
NOTE: make sure to open a server for CoreNLP API from a terminal before running
this script; else, it will crash at the tokenization step. Run with:

java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload
tokenize,ssplit -status_port 9000 -port 9000 -timeout 15000 &

See https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK for details.
"""


def doc_to_tokens_generator(paths):

    n_docs = len(paths)
    for i, doc in enumerate(mp.Pool().imap_unordered(tokenize, paths)):
        print(i, ', ', n_docs, end='\r')
        yield doc


def tokenize(doc):
    try:
        tokenizer = CoreNLPParser()
        with open(doc) as f:
            text = f.read()
        tokens = list(tokenizer.tokenize(text))
        return tokens
    except requests.exceptions.HTTPError as e:
        print('HTTPError for:', doc)
        return []





# prepare data generator
doc_paths = glob.glob(datapaths.PATH_TO_PMC + '/PMC002XXXXXX.txt/**/*.txt',
                      recursive=True)

start = timeit.default_timer()
docs_to_tokens = doc_to_tokens_generator(doc_paths[:10000])

# prepare processors
stats = CorpusStats(docs_to_tokens, max_ngram=3, n_docs=10000)
end = timeit.default_timer()
print('Time elapsed:', end - start)




