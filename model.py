import sys # import nest later, so it won't display its startup message if we fail on config
from scipy import stats

# Global config.
prm = {
        'max_text_len': 6,
        'letter_focus_time': 50.0,
        'readings_path': 'readings/',

        'neuron_type': 'iaf_psc_alpha',
        'letter_neuron_params_on': { 'I_e': 376.0 }, # constant input current in pA
        'letter_column_size': 4,
        'head_column_size': 12,
        'channel_column_size': 8,
        'grapheme_column_size': 8,
        # Synapse specifications.
        'letter_col_lateral_inhibition': { 'weight': -100.0 },
        'letter_head_excitation': { 'weight': 300.0 },
        'head_grapheme_excitation': { 'weight': 300.0 },
        'member_letter_excitation': { 'weight': 0.0 },#4.0 },
        'absent_letter_inhibition': { 'weight': 0.0 },#-4.0 },
        'shorter_word_inhibition': { 'weight': 0.0 },#-1.2 },
        # weights letter -> heights are divided by (1 + (grapheme_len-1)*this)
        'grapheme_length_damping': 0.5,

        'letters': ['a', 'e', 'l', 's', 't', 'u', 'z'],
        'graphemes': ['a', 'e', 'ae', 'ea', 'ee', 'l', 'll', 's', 't', 'u', 'z'],
        'vocabulary': [
            'else',
            'lease',
            'least',
            'lute',
            'sell',
            'sells',
            'set',
            'stu',
            'tall',
            'tell',
            'tells',
            'zet',
            'zest' ]
        }
graphemes_by_lengths = [[g for g in prm['graphemes'] if len(g) == l]
                        for l in range(max([len(g) for g in prm['graphemes']])+1)]

# Parse command line args.
if not (len(sys.argv) in [2, 3]):
    print('Usage: model.py TEXT_INPUT [EXPERIMENT_NAME]')
    sys.exit(-1)
if len(sys.argv[1]) > prm['max_text_len']:
    print('Usage: model.py TEXT_INPUT [EXPERIMENT_NAME]')
    print('TEXT_INPUT has to be shorter than max_text_len: {}'.format(prm['max_text_len']))
    sys.exit(-1)
net_text_input = sys.argv[1]

experiment_name = 'experim'
if len(sys.argv) == 3:
    experiment_name = sys.argv[2]
simulation_name = experiment_name + '_' + net_text_input

# Build the network.
import nest
from neuro_reporting import insert_probe, write_readings

def make_hypercolumn(stimuli_set, column_size):
    return dict([(s, nest.Create(prm['neuron_type'], column_size)) for s in stimuli_set])

lexical_cols = dict([(w, nest.Create(prm['neuron_type'])) for w in prm['vocabulary']])
letter_hypercolumns = [make_hypercolumn(prm['letters'], prm['letter_column_size'])
                       for i in range(prm['max_text_len'])]
# Reading heads' columns are sorted in separate lists by grapheme lengths.
reading_head = [make_hypercolumn(size_graphemes, prm['head_column_size'])
                for size_graphemes in graphemes_by_lengths]
grapheme_hypercolumns = [make_hypercolumn(prm['graphemes'], prm['grapheme_column_size'])
                         for i in range(prm['max_text_len'])]

# Make connections.
for (hcol_n, hypercol) in enumerate(letter_hypercolumns): # hypercol is: letter -> (neuron's nest id)
    # Turn on appropriate letter columns.
    if hcol_n < len(net_text_input) and net_text_input[hcol_n] in prm['letters']:
        nest.SetStatus(hypercol[net_text_input[hcol_n]], prm['letter_neuron_params_on'])

    # Letter hypercol's lateral inhibition to subsequent hypercols
    for hypercol2 in letter_hypercolumns[hcol_n+1:]:
        nest.Connect(sum([list(col) for col in hypercol.values()], []),
                     sum([list(col) for col in hypercol2.values()], []),
                     syn_spec=prm['letter_col_lateral_inhibition'])
    # Letter hypercol -> the reading head
    for (letter, letter_col) in hypercol.items():
        for len_graphemes in reading_head:
            for (grapheme, grapheme_col) in len_graphemes.items():
                if letter in grapheme:
                    nest.Connect(letter_col, grapheme_col, syn_spec=prm['letter_head_excitation'])
    # Letter hypercol -> lexical units
    for (word, word_col) in lexical_cols.items():
        if hcol_n >= len(word):
            nest.Connect(sum([list(col) for col in hypercol.values()], []),
                         word_col,
                         syn_spec=prm['shorter_word_inhibition'])
        else:
            for (letter, letter_col) in hypercol.items():
                if letter in word:
                    nest.Connect(letter_col, word_col, syn_spec=prm['member_letter_excitation'])
                else:
                    nest.Connect(letter_col, word_col, syn_spec=prm['absent_letter_inhibition'])
for len_graphemes in reading_head:
    for (grapheme, grapheme_col) in len_graphemes.items():
        nest.Connect(grapheme_col,
                     sum([list(neurs)
                          for col in grapheme_hypercolumns
                          for (label, neurs) in col.items()
                          if label == grapheme], []),
                     syn_spec=prm['head_grapheme_excitation'])

# Insert probes:
###for (word, word_col) in lexical_cols.items():
###    insert_probe(word_col, word)
for (letter, letter_col) in letter_hypercolumns[1].items():
    insert_probe(letter_col, 'L2-'+letter)
for len_graphemes in reading_head:
    for (grapheme, grapheme_col) in len_graphemes.items():
        insert_probe(grapheme_col, 'head-'+grapheme)
for (grapheme, grapheme_col) in grapheme_hypercolumns[0].items():
    insert_probe(grapheme_col, 'G1-'+grapheme)
for (grapheme, grapheme_col) in grapheme_hypercolumns[1].items():
    insert_probe(grapheme_col, 'G2-'+grapheme)

# Run the simulation, write readings.
nest.Simulate(prm['letter_focus_time'])
with open('wgts', 'w+') as out:
    for lett_n in range(prm['max_text_len']):
        # Reassign the letter -> head weights.
        weights_dist = stats.skewnorm(4, loc=lett_n-0.7, scale=2)
        print(lett_n, ':', file=out)
        print(weights_dist.pdf(range(prm['max_text_len'])) * 5000, file=out)
        for assg_lett_n in range(prm['max_text_len']):
            assg_hypercol = sum([list(col) for (lett, col) in letter_hypercolumns[assg_lett_n].items()], [])
            for (ln, len_graphemes) in enumerate(reading_head):
                len_graphemes = sum([list(col) for (lett, col) in len_graphemes.items()], [])
                if len(len_graphemes) == 0:
                    continue
                nest.SetStatus( nest.GetConnections(assg_hypercol, len_graphemes),
                                { 'weight' : (  weights_dist.pdf(assg_lett_n) * 5000
                                              / (1.0 + (ln-1)*prm['grapheme_length_damping'])) })
        nest.Simulate(prm['letter_focus_time'])

write_readings(prm['readings_path']+simulation_name,
               params=prm,
               spike_groups={ 'Head': ['head-'+g for g in prm['graphemes']] })#, spike_groups={ 'words': prm['vocabulary'] })
