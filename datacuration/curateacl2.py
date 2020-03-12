import os
import re

################################################################################
# The ACL 2.0 corpus comes in some different folders. All information, however,
# is contained in the "vertical files" (.vert) with sentence, token, POS-tag and
# term annotations given on lines per token. This script splits those files into
# single documents.
################################################################################

# resolve in and out files
path_in = input('Please, input the full path to the ACL .vert file: ')
path_out = input('Please, input the full path to the folder where the split ' +
                 'files should be saved: ')
try:
    os.chdir(path_out)
except FileNotFoundError:
    print('The folder does not exist; I\'ll create it for you!')
    os.mkdir(path_out)
    os.chdir(path_out)

used_ids = set()
with open(path_in) as in_file:
    current_doc = ''
    doc_id = ''
    for line in in_file:
        current_doc += line
        if line.strip().startswith('<doc'):  # new document
            match = re.match('<doc id="(.*?)"', line)
            doc_id = match.group(1)
            if doc_id not in used_ids:
                used_ids.add(doc_id)
                doc_id += '_1'
            else:
                doc_id += '_2'
        if line.strip().startswith('</doc'):
            with open(doc_id + '.vert', 'w+') as out:
                print(current_doc, file=out)
            current_doc = ''
            doc_id = ''

    print('Done!')
