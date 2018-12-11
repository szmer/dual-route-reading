# Read language data.

def read_lang(dir_path, stems_and_prefixes=False):
    """Load letters, graphemes, vocabulary, suffixes from the dir_path and
    return as lists, except for the vocabulary, which is returned as a dict
    sorted by initial letters."""
    letters, graphemes, vocabulary = None, None, dict()
    with open(dir_path+'letters') as fl:
        letters = fl.read().strip().split()
    with open(dir_path+'graphemes') as fl:
        graphemes = fl.read().strip().split()
    # Vocabulary is sorted by the first letter.
    skipped_words_n = 0
    vocabulary_path = dir_path + ('stems' if stems_and_suffixes else 'vocabulary')
    with open(vocabulary_path) as fl:
        for line in fl:
            line = line.strip()
            if [lett for lett in line if not lett in letters]:
                skipped_words_n += 1
                continue
            index_lett = unidecode(line[0])
            if not index_lett in vocabulary:
                vocabulary[index_lett] = []
            vocabulary[index_lett].append(line)
    print('{} vocabulary words skipped (unknown letters present)'.format(skipped_words_n))
    # Suffixes are saved only as lists of their graphemes.
    suffixes = []
    if stems_and_suffixes:
        with open(dir_path + 'suffixes') as fl:
            for line in fl:
                suffix = line.strip()
                suffixes.append(suffix)

    return letters, graphemes, vocabulary, suffixes
