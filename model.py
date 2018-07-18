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
###        'head_channels_excitation': { 'weight': 160.0 },
###        'channel_lateral_inhibition': { 'weight': -100.0 },
###        'channel_grapheme_excitation': { 'weight': 160.0 },
###        'grapheme_letter_inhibition': { 'weight': -200.0 },
###        'grapheme_channel_inhibition': { 'weight': -250.0 },
        'member_letter_excitation': { 'weight': 0.0 },#4.0 },
        'absent_letter_inhibition': { 'weight': 0.0 },#-4.0 },
        'shorter_word_inhibition': { 'weight': 0.0 },#-1.2 },

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
reading_head = make_hypercolumn(prm['graphemes'], prm['head_column_size'])
###grapheme_channels = [make_hypercolumn(prm['graphemes'], prm['channel_column_size'])
###                     for i in range(prm['max_text_len'])]
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
        for (grapheme, grapheme_col) in reading_head.items():
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
for (grapheme, grapheme_col) in reading_head.items():
    nest.Connect(grapheme_col,
                 sum([list(neurs)
                      for col in grapheme_hypercolumns
                      for (label, neurs) in col.items()
                      if label == grapheme], []),
                 syn_spec=prm['head_grapheme_excitation'])
###                      for channel in grapheme_channels
###                      for (label, col) in channel.items()
###                      if label == grapheme], []),
###                 syn_spec=prm['head_channels_excitation'])
###for (chan_n, chan) in enumerate(grapheme_channels):
###    for chan2 in grapheme_channels[chan_n+1:]:
###        nest.Connect(sum([list(col) for col in chan.values()], []),
###                     sum([list(col) for col in chan2.values()], []),
###                     syn_spec=prm['channel_lateral_inhibition'])
###    for (grapheme, grapheme_col) in chan.items():
###        nest.Connect(grapheme_col, grapheme_hypercolumns[chan_n][grapheme],
###                     syn_spec=prm['channel_grapheme_excitation'])
###for (hcol_n, hypercol) in enumerate(grapheme_hypercolumns):
###    hypercol = sum([list(col) for col in hypercol.values()], [])
###    nest.Connect(hypercol, sum([list(col) for col in letter_hypercolumns[hcol_n].values()], []),
###                 syn_spec=prm['grapheme_letter_inhibition'])
###    nest.Connect(hypercol, sum([list(col) for col in grapheme_channels[hcol_n].values()], []),
###                 syn_spec=prm['grapheme_channel_inhibition'])

# Insert probes:
###for (word, word_col) in lexical_cols.items():
###    insert_probe(word_col, word)
for (letter, letter_col) in letter_hypercolumns[1].items():
    insert_probe(letter_col, 'L2-'+letter)
for (grapheme, grapheme_col) in reading_head.items():
    insert_probe(grapheme_col, 'head-'+grapheme)
###for (grapheme, grapheme_chan) in grapheme_channels[0].items():
###    insert_probe(grapheme_chan, 'C1-'+grapheme)
###for (grapheme, grapheme_chan) in grapheme_channels[1].items():
###    insert_probe(grapheme_chan, 'C2-'+grapheme)
for (grapheme, grapheme_col) in grapheme_hypercolumns[0].items():
    insert_probe(grapheme_col, 'G1-'+grapheme)
for (grapheme, grapheme_col) in grapheme_hypercolumns[1].items():
    insert_probe(grapheme_col, 'G2-'+grapheme)

# Run the simulation, write readings.
for lett_n in range(prm['max_text_len']):
    # Reassign the letter -> head weights.
    weights_dist = stats.skewnorm(4, loc=float(lett_n)-0.7, scale=2)
    for lett_w_n in range(prm['max_text_len']):
        nest.SetStatus(
            nest.GetConnections(sum([list(col) for (lett, col) in letter_hypercolumns[lett_w_n].items()], []),
                                sum([list(col) for (lett, col) in reading_head.items()], [])),
            { 'weight' : weights_dist.pdf(lett_w_n) * 5000 })
    nest.Simulate(prm['letter_focus_time'])

write_readings(prm['readings_path']+simulation_name,
               params=prm,
               spike_groups={ 'Head': ['head-'+g for g in prm['graphemes']] })#, spike_groups={ 'words': prm['vocabulary'] })
