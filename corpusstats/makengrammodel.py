import colibricore
from datautils import dataio as dio, annotations as anno
from tqdm import tqdm

TEMP = '/home/kasper/Desktop/'

def tokenized_text_from_doc(doc: anno.Document):
    string_builder = ''
    for sent in doc.get_annotations('Sentence'):
        tokenized_sent = ' '.join(
            t.get_covered_text() + '/' + t.mapped_pos()
            for t in doc.get_annotations_at(sent.span, 'Token')
        )
        string_builder += tokenized_sent + '\n'
    return string_builder


corpus = dio.pmc_corpus_ids()
corpus_text = ''.join(tqdm((tokenized_text_from_doc(dio.load_pmc_document(doc))
                           for doc in corpus), total=len(corpus)))
corpus_plain_text_file = TEMP + 'genia_tokenized.txt'
with open(corpus_plain_text_file, 'w+') as out_file:
    out_file.write(corpus_text)

class_file = TEMP + 'genia.colibri.cls'
encoder = colibricore.ClassEncoder()
encoder.build(corpus_plain_text_file)
encoder.save(class_file)
#
encoded_corpus_file = TEMP + 'genia.colibri.dat'
encoder.encodefile(corpus_plain_text_file, encoded_corpus_file)

decoder = colibricore.ClassDecoder(class_file)

corpus_index = colibricore.IndexedCorpus(encoded_corpus_file)

model_options = colibricore.PatternModelOptions(mintokens=2, maxlength=7, doskipgrams=True)
model = colibricore.IndexedPatternModel(reverseindex=corpus_index)
model.train(encoded_corpus_file, model_options)

