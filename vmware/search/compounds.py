from .freqs import freq_per_term, freq_per_phrase
from vmware.index.colocations import index, queries
import pickle


def get_most_freq_compound_dicts():
    # print("From Queries")
    colocs = queries
    to_decompound = {}   # Compounds -> decompounded forms
    to_compound = set()  # list of all forms decompounded from compounds

    for row in colocs[colocs['compound_count'] > 0].to_dict(orient='records'):
        compound_freq = freq_per_term(row['first_term'] + row['second_term'])
        decompound_freq = freq_per_phrase([row['first_term'] + " " + row['second_term']])
        assert len(compound_freq.values()) == 1
        assert len(decompound_freq.values()) == 1
        compound_freq = list(compound_freq.values())[0]
        decompound_freq = list(decompound_freq.values())[0]
        # print(row['first_term'] + " " + row['second_term'],
        #      compound_freq, decompound_freq)
        if compound_freq < decompound_freq:
            # What we want to DECOMPOUND
            to_decompound[row['first_term'] + row['second_term']] = \
                row['first_term'] + " " + row['second_term']
        else:
            to_compound.add((row['first_term'], row['second_term']))
    return to_decompound, to_compound


most_freq_compound_strategy = get_most_freq_compound_dicts()


def get_most_freq_corpus_compound_dicts():
    colocs = index
    to_decompound = {}   # Compounds -> decompounded forms
    to_compound = set()  # list of all forms decompounded from compounds
    for row in colocs[colocs['compound_count'] > 0].to_dict(orient='records'):
        compound_freq = freq_per_term(row['first_term'] + row['second_term'])
        decompound_freq = freq_per_phrase([row['first_term'] + " " + row['second_term']])
        assert len(compound_freq.values()) == 1
        assert len(decompound_freq.values()) == 1
        compound_freq = list(compound_freq.values())[0]
        decompound_freq = list(decompound_freq.values())[0]
        # print(row['first_term'] + " " + row['second_term'],
        #      compound_freq, decompound_freq)
        if compound_freq < decompound_freq:
            # What we want to DECOMPOUND
            to_decompound[row['first_term'] + row['second_term']] = \
                row['first_term'] + " " + row['second_term']
        else:
            to_compound.add((row['first_term'], row['second_term']))
    return to_decompound, to_compound


def to_compound_query(query, to_decompound, to_compound):
    new_query = []
    last_term = ''
    fast_forward = False
    for first_term, second_term in zip(query.split(), query.split()[1:]):
        last_term = second_term
        if fast_forward:
            print("Skipping: " + first_term + " " + second_term)
            fast_forward = False
            continue
        first_term = first_term.strip().lower()
        second_term = second_term.strip().lower()
        if first_term in to_decompound:
            new_query.append(to_decompound[first_term])
        elif (first_term, second_term) in to_compound:
            new_query.append(first_term + second_term)
            fast_forward = True
        else:
            new_query.append(first_term)
    if not fast_forward:
        if last_term in to_decompound:
            new_query.append(to_decompound[last_term])
        else:
            new_query.append(last_term)
    return new_query


def get_corpus_and_query_compound_dicts():
    most_freq_compound_query_strategy = get_most_freq_compound_dicts()
    most_freq_compound_corpus_strategy = get_most_freq_corpus_compound_dicts()

    to_decompound = {**most_freq_compound_query_strategy[0], **most_freq_compound_corpus_strategy[0]}
    to_compound = set(most_freq_compound_query_strategy[1]) | set(most_freq_compound_corpus_strategy[1])
    return to_decompound, to_compound


try:
    most_freq_compound_strategy = pickle.load(open("data/cache/most_freq_compound_strategy.pkl", "rb"))
except FileNotFoundError:
    most_freq_compound_strategy = get_most_freq_compound_dicts()
    pickle.dump(most_freq_compound_strategy, open("data/cache/most_freq_compound_strategy.pkl", "wb"))

try:
    most_freq_compound_corpus_strategy = pickle.load(open("data/cache/most_freq_compound_corpus_strategy.pkl", "rb"))
except FileNotFoundError:
    most_freq_compound_corpus_strategy = get_most_freq_corpus_compound_dicts()
    pickle.dump(most_freq_compound_corpus_strategy, open("data/cache/most_freq_compound_corpus_strategy.pkl", "wb"))

try:
    most_freq_compound_both_strategy = pickle.load(open("data/cache/most_freq_compound_both_strategy.pkl", "rb"))
except FileNotFoundError:
    most_freq_compound_both_strategy = get_corpus_and_query_compound_dicts()
    pickle.dump(most_freq_compound_both_strategy, open("data/cache/most_freq_compound_both_strategy.pkl", "wb"))
