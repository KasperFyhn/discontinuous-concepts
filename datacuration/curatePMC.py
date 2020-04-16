from tqdm import tqdm
import os
import shutil

count_file_path = input('Path to count file: ')
threshold = int(input('Minimum number of concepts in doc: '))
pmc_docs_dir = input('Path to all PMC documents: ')
output_dir = input('Path to dir with curated documents: ')

with open(count_file_path) as count_file:
    concept_counts = {line.split()[0]: int(line.split()[2])
                      for line in count_file}

ranked = sorted(concept_counts.keys(), key=lambda key: concept_counts[key],
                reverse=True)
best_docs = []
for doc in ranked:
    if concept_counts[doc] > threshold:
        best_docs.append(doc)
    else:
        break

print(len(best_docs), 'documents have more than', threshold, 'concepts.')

for doc_id in tqdm(best_docs, desc='Copying files'):
    try:
        # the documents are stored in nested folders,
        # e.g. PMC/PMC001XXXXXX.txt/PMC0012XXXXX/PMC1249490.txt
        # build this path based on the provided ID
        first_folder = 'PMC00' + doc_id[3] + 'XXXXXX.txt/'
        second_folder = 'PMC00' + doc_id[3:5] + 'XXXXX/'
        text_file = first_folder + second_folder + doc_id + '.txt'
        source_file = os.path.join(pmc_docs_dir, text_file)
        destination_file = os.path.join(output_dir, text_file)
        destination_dir = os.path.dirname(destination_file)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        shutil.copyfile(source_file, destination_file)

        anno_file = os.path.join('annotations', doc_id[:4], doc_id + '.anno')
        annotations_source = os.path.join(pmc_docs_dir, anno_file)
        annotations_destination = os.path.join(output_dir, anno_file)
        destination_dir = os.path.dirname(annotations_destination)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        shutil.copyfile(annotations_source, annotations_destination)
    except Exception as e:
        print(doc_id, 'could not be copied!')
        print(e)

print('Done!')

