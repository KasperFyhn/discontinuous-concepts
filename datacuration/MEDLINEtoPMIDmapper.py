import glob
from bs4 import BeautifulSoup
from datautils import datapaths

################################################################################
# The purpose of this script is to map ID's with the format MEDLINE:12345678 to
# PMID's since documents from GENIA in pos+concepts carry the first ID while
# documents in treebank carry the latter.
################################################################################

# load pos+concepts articles and normalize the titles
pos_concepts = {}
for file in glob.glob(datapaths.PATH_TO_GENIA + '/pos+concepts/*.xml'):
    soup = BeautifulSoup(open(file).read(), 'xml')
    doc_id = soup.bibliomisc.string[8:]
    title = ''.join(word for word in soup.title.strings) \
        .strip().replace(' ', '').replace('\n', '')
    pos_concepts[title] = doc_id

# load treebank articles and normalize the titles
treebank = {}
for file in glob.glob(datapaths.PATH_TO_GENIA + '/treebank/*.xml'):
    soup = BeautifulSoup(open(file).read(), 'xml')
    doc_id = soup.PMID.string
    title = ''.join(word for word in soup.ArticleTitle.strings)\
        .strip().replace(' ', '').replace('\n', '')
    treebank[title] = doc_id

# map MEDLINE id's to PMID's through the titles
mapping = {medline: treebank[title]
           for title, medline in pos_concepts.items()}

# save mapping
with open(datapaths.PATH_TO_GENIA + '/MEDLINE-to-PMID', 'w+') as out:
    print(mapping, file=out)

