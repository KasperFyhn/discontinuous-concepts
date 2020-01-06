import sys
from datautils import dataio
import os
from colorama import Style, Back, Fore

DEBUG = False
if DEBUG:  # DEBUG MODE
    print('Running debug with: GENIA 90208323 Concept Constituent POS')
    sys.argv.append('GENIA')
    sys.argv.append('90208323')
    sys.argv.append('Concept')
    sys.argv.append('Constituent')
    sys.argv.append('POS')

assert len(sys.argv) > 3, (
        "The script should be run with arguments: "
        + "CORPUS DOC_ID ANNOTATION_TYPE [SECOND_TYPE] [pos]"
)


def get_doc_text_chunk(start, end, with_pos_tags=False):

    if with_pos_tags:
        tokens = doc.get_annotations('Token')
        tokens_in_chunk = [token for token in tokens
                           if start <= token.span[0] and token.span[1] <= end]
        chunk = ' '.join(token.get_covered_text() + '\\' + token.pos
                         for token in tokens_in_chunk)
        return chunk

    else:
        return DOC_TEXT[start:end]


COLOR_BEGIN = Back.CYAN + Fore.BLACK
COLOR_END = Style.RESET_ALL
WINDOW = 2

# parse all CL arguments and set things accordingly

# determine which corpus to load from and set function accordingly
if sys.argv[1].lower() == 'genia':
    load_doc = dataio.load_genia_document
elif sys.argv[1].lower() == 'craft':
    load_doc = dataio.load_craft_document

if sys.argv[2].isdigit():
    docs = iter([load_doc(sys.argv[2])])
elif sys.argv[2].startswith('[') and sys.argv[2].endswith(']'):
    docs = iter(load_doc(d) for d in eval(sys.argv[2]))
elif sys.argv[2] == 'ALL' and sys.argv[1].lower() == 'genia':
    docs = iter(dataio.load_genia_corpus())
elif sys.argv[2] == 'ALL' and sys.argv[1].lower() == 'craft':
    docs = iter(dataio.load_craft_corpus())

try:
    if sys.argv[4].lower() == 'pos':
        WITH_POS_TAGS = True
    else:
        WITH_POS_TAGS = False
except IndexError:
    WITH_POS_TAGS = False

try:
    if sys.argv[4].lower() != 'pos':
        secondary_type = sys.argv[4]
    else:
        secondary_type = None
except IndexError:
    secondary_type = None

# get ready to run
doc = next(docs)
DOC_TEXT = doc.get_text()
annotations = doc.get_annotations(sys.argv[3])

current_index = 0

while True:
    if not annotations:
        print(doc.id, 'has no annotations of the requested type!')
        input('Moving on to next document.')
        doc = next(docs)
        DOC_TEXT = doc.get_text()
        annotations = doc.get_annotations(sys.argv[3])
        current_index = 0
        continue

    current_annotation = annotations[current_index]
    if isinstance(current_annotation, dataio.DiscontinuousConcept):
        spans = current_annotation.spans
    else:
        spans = [current_annotation.span]

    para_left = DOC_TEXT.rfind('\n', 0, spans[0][0])
    para_left = para_left if para_left > 0 else 0
    para_right = DOC_TEXT.find('\n', spans[-1][1])
    para_right = para_right if para_left < para_right < len(DOC_TEXT)\
        else len(DOC_TEXT)

    print_text = ''
    at_char = para_left
    for span in spans:
        print_text += get_doc_text_chunk(at_char, span[0],
                                         with_pos_tags=WITH_POS_TAGS) \
                      + (' ' if WITH_POS_TAGS else '')\
                      + COLOR_BEGIN \
                      + get_doc_text_chunk(span[0], span[1],
                                           with_pos_tags=WITH_POS_TAGS)\
                      + COLOR_END \
                      + (' ' if WITH_POS_TAGS else '')

        at_char = span[1]

    # add the last bit
    print_text += get_doc_text_chunk(at_char, para_right,
                                     with_pos_tags=WITH_POS_TAGS)

    left_cut = DOC_TEXT.rfind('\n', 0, para_left)
    left_cut = left_cut if left_cut > 0 else 0
    right_cut = DOC_TEXT.find('\n', para_right + 1)
    right_cut = right_cut if left_cut < right_cut < len(DOC_TEXT) \
        else len(DOC_TEXT)
    for i in range(WINDOW):
        left_cut = DOC_TEXT.rfind('\n', 0, left_cut)
        left_cut = left_cut if left_cut > 0 else 0
        right_cut = DOC_TEXT.find('\n', right_cut + 1)
        right_cut = right_cut if left_cut < right_cut < len(DOC_TEXT)\
            else len(DOC_TEXT)

    print_text = get_doc_text_chunk(left_cut, para_left) \
                 + ('\n' if WITH_POS_TAGS else '') \
                 + print_text \
                 + get_doc_text_chunk(para_right, right_cut)
    print_text = print_text.strip()

    os.system('clear')
    print(print_text)
    print('\n' + '='*50)
    print(f'{COLOR_BEGIN}{current_annotation}{COLOR_END}')
    if secondary_type:
        annotations_at_this = doc.get_annotations_at(current_annotation.span,
                                                     secondary_type)
        for anno in annotations_at_this:
            print(anno)
    print('\nDocument:', doc.id,
          f'\tShowing annotation {current_index+1} of {len(annotations)}')
    choice = input('Enter = next; b = previous; n = next doc; q to quit: ')
    if choice == '':
        current_index += 1
        if current_index > len(annotations) - 1:
            current_index = len(annotations) - 1
    elif choice == 'b':
        current_index -= 1
        if current_index < 0:
            current_index = 0
    elif choice == 'n':
        try:
            doc = next(docs)
        except StopIteration:
            exit('No more documents')
        DOC_TEXT = doc.get_text()
        annotations = doc.get_annotations(sys.argv[3])
        current_index = 0
    elif choice == 'q':
        break

