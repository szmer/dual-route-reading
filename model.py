import sys # import nest later, so it won't display its startup message if we fail on config

# Global config.
max_text_len = 6
letter_perception_time = 2000.0
letter_switch_time = 500.0 # how long after starting to perceive one letter we start to see the next one
simulation_time = letter_switch_time * max_text_len + 2.0 * letter_perception_time
readings_path = 'readings/'

dc_gen_params = { 'amplitude': 5000.0 }
neuron_type = 'iaf_psc_alpha'
letter_column_size = 40
grapheme_column_size = 20
# Synapse specifications.
letter_col_lateral_inhibition = { 'weight': -16.0 }
letter_head_excitation = { 'weight': 6.0 }
member_letter_excitation = { 'weight': 0.0 }#4.0 }
absent_letter_inhibition = { 'weight': 0.0 }#-4.0 }
shorter_word_inhibition = { 'weight': 0.0 }#-1.2 }

letters = ['a', 'e', 'l', 's', 't', 'u', 'z']
graphemes = ['a', 'e', 'ae', 'ea', 'ee', 'l', 'll', 's', 't', 'u', 'z']
vocabulary = [
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

# Parse command line args.
if not (len(sys.argv) in [2, 3]):
    print('Usage: model.py TEXT_INPUT [EXPERIMENT_NAME]')
    sys.exit(-1)
if len(sys.argv[1]) > max_text_len:
    print('Usage: model.py TEXT_INPUT [EXPERIMENT_NAME]')
    print('TEXT_INPUT has to be shorter than max_text_len: {}'.format(max_text_len))
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
    return dict([(s, nest.Create(neuron_type, column_size)) for s in stimuli_set])

lexical_cols = dict([(w, nest.Create(neuron_type)) for w in vocabulary])
letter_hypercolumns = [make_hypercolumn(letters, letter_column_size)
                       for i in range(max_text_len)]
reading_head = make_hypercolumn(graphemes, grapheme_column_size)
grapheme_channels = [make_hypercolumn(graphemes, grapheme_column_size)
                     for i in range(max_text_len)]
grapheme_hypercolumns = [make_hypercolumn(graphemes, grapheme_column_size)
                         for i in range(max_text_len)]

dc_gens = nest.Create('dc_generator', max_text_len, params=dc_gen_params)
for (dcg_n, dc_gen) in enumerate(dc_gens): # both are numbers, but dc_gen in the sequence of all nest objects
    nest.SetStatus([dc_gen], { 'start': dcg_n*letter_switch_time,
                               'stop': letter_perception_time + dcg_n*letter_switch_time })

# Make connections.
for (hcol_n, hypercol) in enumerate(letter_hypercolumns): # hypercol is: letter -> (neuron's nest id)
    # DC generator -> letter hypercol (the cell corresponding to the letter at this position)
    if hcol_n < len(net_text_input) and net_text_input[hcol_n] in letters:
        nest.Connect([dc_gens[hcol_n]], hypercol[net_text_input[hcol_n]])
    # Letter hypercol's lateral inhibition to subsequent hypercols
    for hypercol2 in letter_hypercolumns[hcol_n+1:]:
        nest.Connect(sum([list(col) for col in hypercol.values()], []),
                     sum([list(col) for col in hypercol2.values()], []),
                     syn_spec=letter_col_lateral_inhibition)
    # Letter hypercol -> the reading head
    for (letter, letter_col) in hypercol.items():
        for (grapheme, grapheme_col) in reading_head.items():
            if letter in grapheme:
                nest.Connect(letter_col, grapheme_col, syn_spec=letter_head_excitation)
    # Letter hypercol -> lexical units
    for (word, word_col) in lexical_cols.items():
        if hcol_n >= len(word):
            nest.Connect(sum([list(col) for col in hypercol.values()], []),
                         word_col,
                         syn_spec=shorter_word_inhibition)
        else:
            for (letter, letter_col) in hypercol.items():
                if letter in word:
                    nest.Connect(letter_col, word_col, syn_spec=member_letter_excitation)
                else:
                    nest.Connect(letter_col, word_col, syn_spec=absent_letter_inhibition)

# Insert probes:
###for (word, word_col) in lexical_cols.items():
###    insert_probe(word_col, word)
for (letter, letter_col) in letter_hypercolumns[1].items():
    insert_probe(letter_col, '2-'+letter)
for (grapheme, grapheme_col) in reading_head.items():
    insert_probe(grapheme_col, 'head-'+grapheme)

# Run the simulation, write readings.
nest.Simulate(simulation_time)
write_readings(readings_path+simulation_name,
               spike_groups={ 'Head': ['head-'+g for g in graphemes] })#, spike_groups={ 'words': vocabulary })
