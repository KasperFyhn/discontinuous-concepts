import nltk
import requests
import multiprocessing as mp
from datautils import datapaths
import datautils.annotations as anno
import glob
import os
from tqdm import tqdm

################################################################################
# This scripts runs over one ore more folders from the PMC corpus and makes NLP
# on each document. Currently, the script performs: Tokenization, sentence
# splitting and POS-tagging.
# Furthermore, it removes the license statement from the top of each document.
################################################################################

# NOTE: make sure to open a server for CoreNLP API from a terminal before
# running this script; else, it will crash
# cf. https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK


LICENSE_STRING = 'LICENSE: This file is available for text mining. It may '\
                 'also be used consistent with the principles of fair use '\
                 'under the copyright law.'

SAVE_DIR = datapaths.PATH_TO_PMC + 'annotations/'


def process(path):
    # load text, remove license statement and make id
    with open(path, encoding='utf-8') as file:
        text = file.read().replace(LICENSE_STRING, '').strip()
    doc_id = os.path.basename(path)[:-4]

    # process document
    parser = nltk.CoreNLPParser(tagtype='pos')
    try:
        annotations = parser.api_call(
            text, properties={'annotators': 'tokenize,ssplit,pos'}
        )
    except requests.exceptions.HTTPError as e:
        # some docs cause such an error; if so, move it out of the way for later
        print('\rHTTPError for:', path)
        new_path = datapaths.PATH_TO_PMC + 'problematic/'\
                   + os.path.basename(path)
        os.rename(path, new_path)
        return

    # prepare Document object to add the annotations to
    doc = anno.Document(doc_id, text)

    # loop over sentences and resolve the spans
    sentences = annotations['sentences']
    for sentence in sentences:
        # make sentence annotation
        tokens = sentence['tokens']
        sentence_begin = tokens[0]['characterOffsetBegin']
        sentence_end = tokens[-1]['characterOffsetEnd']
        sent_anno = anno.Sentence(doc, (sentence_begin, sentence_end))
        doc.add_annotation(sent_anno)

        # loop over tokens to make token annotations
        for token in tokens:
            token_begin = token['characterOffsetBegin']
            token_end = token['characterOffsetEnd']
            pos_tag = token['pos']
            token_anno = anno.Token(doc, (token_begin, token_end), pos_tag)
            doc.add_annotation(token_anno)

    # correct the text in the file
    with open(path, 'w', encoding='utf-8') as new_file:
        print(text, file=new_file)

    # save annotations from Document object
    doc.save_annotations_to_file(SAVE_DIR)


if __name__ == '__main__':
    # load doc paths
    doc_paths = glob.glob(datapaths.PATH_TO_PMC + '/PMC00*XXXXXX.txt/**/*.txt',
                          recursive=True)

    # loop over doc paths and process the documents one by one (but in parallel)
    for doc_path in tqdm(mp.Pool().imap_unordered(process, doc_paths),
                         total=len(doc_paths)):
        continue




