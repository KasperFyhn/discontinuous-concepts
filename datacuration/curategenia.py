import os
from bs4 import BeautifulSoup

################################################################################
# The GENIA corpus comes as one big XML file.
# The purpose of this script is simply to cut it into smaller files with their
# appropriate ID's as file names.
################################################################################

# resolve in and out files
path_in = input('Please, input the full path to the GENIA set file: ')
xml_file = open(path_in)
path_out = input('Please, input the full path to the folder where the split ' +
                 'files should be saved: ')
try:
    os.chdir(path_out)
except FileNotFoundError:
    print('The folder does not exist; I\'ll create it for you!')
    os.mkdir(path_out)
    os.chdir(path_out)

# read XML; might take some time
print('Loading XML file ...')
bs = BeautifulSoup(xml_file.read(), 'xml')

# write files in a more uniform format
print('Writing files ...')
for article in bs.find_all('article'):
    doc_id = article.bibliomisc.string[8:]
    with open(doc_id + '.xml', 'w+') as out:
        print('<?xml version="1.0" encoding="UTF-8"?>\n\n', article, sep='',
              file=out)

print('Done!')
