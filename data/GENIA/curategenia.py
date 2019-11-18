import os
from bs4 import BeautifulSoup

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

print('Loading XML file ...')
bs = BeautifulSoup(xml_file.read(), 'xml')

print('Writing files ...')
for article in bs.find_all('article'):
    doc_id = article.bibliomisc.string[8:]
    with open(doc_id + '.xml', 'w+') as out:
        print('<?xml version="1.0" encoding="UTF-8"?>\n\n', article, sep='',
              file=out)

print('Done!')
