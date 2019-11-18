import glob
from bs4 import BeautifulSoup

"""The purpose of this script is to map ID's with the format MEDLINE:12345678 to
PMID's since documents in pos+concepts carry the first ID while documents in
treebank carry the latter."""

pos_concepts = {}
for file in glob.glob('./pos+concepts/*.xml'):
    soup = BeautifulSoup(open(file).read(), 'xml')
    doc_id = soup.bibliomisc.string[8:]
    title = ''.join(word for word in soup.title.strings) \
        .strip().replace(' ', '').replace('\n', '')
    pos_concepts[title] = doc_id

treebank = {}
for file in glob.glob('./treebank/*.xml'):
    soup = BeautifulSoup(open(file).read(), 'xml')
    doc_id = soup.PMID.string
    title = ''.join(word for word in soup.ArticleTitle.strings)\
        .strip().replace(' ', '').replace('\n', '')
    treebank[title] = doc_id

mapping = {medline: treebank[title]
           for title, medline in pos_concepts.items()}

with open('MEDLINE-to-PMID', 'w+') as out:
    print(mapping, file=out)

