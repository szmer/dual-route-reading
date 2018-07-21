from scipy import stats
import nest
from neuro_reporting import reset_reporting, insert_probe, write_readings, decide_spikes

nest.set_verbosity('M_WARNING') # don't print detailed simulation info

def decompose_word(word):
    "Get a list of graphemes in the word."
    current_pos = 0
    graphemes = []
    while current_pos < len(word):
        found = False
        for len_graphemes in reversed(graphemes_by_lengths):
            for grapheme in len_graphemes:
                if word[current_pos:current_pos+len(grapheme)] == grapheme:
                    graphemes.append(grapheme)
                    found = True
                    break
            if found:
                break
        else:
            raise ValueError('Cannot decompose string {} in word {}'.format(word[current_pos:], word))
        current_pos += len(graphemes[-1])
    return graphemes

def make_hypercolumn(stimuli_set, column_size):
    return dict([(s, nest.Create(prm['neuron_type'], column_size)) for s in stimuli_set])

# Global config.
prm = {
        'max_text_len': 6,
        'letter_focus_time': 50.0,
        'readings_path': 'readings/',

        'neuron_type': 'iaf_psc_alpha',
        'letter_neuron_params_on': { 'I_e': 900.0 }, # constant input current in pA
        'letter_column_size': 4,
        'head_column_size': 12,
        'grapheme_column_size': 8,
        'lexical_column_size': 1,
        'lexical_inhibiting_pop_size': 10,
        # Synapse specifications.
        'letter_col_lateral_inhibition': { 'weight': -100.0 },
        'letter_head_excitation': { 'weight': 300.0 },
        'member_first_letter_excitation': { 'weight': 190.0 },
        'member_letter_excitation_weight': 400.0, # make them separate so they show up in readings printouts
        'member_letter_inhibition_weight': -400.0,
        'member_letter_excitation': (lambda length: { 'weight': prm['member_letter_excitation_weight'] / length }),
        'absent_letter_inhibition': (lambda length: { 'weight': prm['member_letter_inhibition_weight'] / length }),
        'shorter_word_inhibition': { 'weight': -200.0 },
        'lexical_grapheme_excitation': { 'weight': 420.0 },
        'lexical_inhibiting_pop_excitation': { 'weight': 200.0 },
        'lexical_inhibiting_pop_feedback': { 'weight': -300.0 },
        # weights letter -> heights are divided by (1 + (grapheme_len-1)*this)
        'grapheme_length_damping': 0.8,
        'head_grapheme_synapse_model': { 'U': 0.67, 'u': 0.67, 'x': 1.0, 'tau_rec': 50.0,
                                        'tau_fac': 0.0, 'weight': 300.},
        # (the _model part in name is meant to mark that we register a separate synampse 'type')

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
spike_groups, spike_decisions = {}, {} # to be filled when preparing a simulation

def simulate_reading(net_text_input):
    if len(net_text_input) > prm['max_text_len']:
        raise ValueError('Text input {} has to be shorter than max_text_len: {}'.format(net_text_input, prm['max_text_len']))

    # Build the network.
    nest.ResetKernel()
    reset_reporting()

    nest.CopyModel('tsodyks2_synapse', 'head_grapheme_synapse_model', prm['head_grapheme_synapse_model'])

    lexical_cols = dict([(w, nest.Create(prm['neuron_type'], prm['lexical_column_size']))
                         for w in prm['vocabulary']])
    lexical_inhibiting_population = nest.Create(prm['neuron_type'], prm['lexical_inhibiting_pop_size'])
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
                    if hcol_n == 0 and word[0] == letter:
                        nest.Connect(letter_col, word_col, syn_spec=prm['member_first_letter_excitation'])
                    elif letter in word:
                        nest.Connect(letter_col, word_col, syn_spec=prm['member_letter_excitation'](len(word)))
                    else:
                        nest.Connect(letter_col, word_col, syn_spec=prm['absent_letter_inhibition'](len(word)))
    nest.Connect(sum([list(col) for (word, col) in lexical_cols.items()], []),
                 lexical_inhibiting_population, syn_spec=prm['lexical_inhibiting_pop_excitation'])
    nest.Connect(lexical_inhibiting_population,
                 sum([list(col) for (word, col) in lexical_cols.items()], []),
                 syn_spec=prm['lexical_inhibiting_pop_feedback'])
    for len_graphemes in reading_head:
        for (grapheme, grapheme_col) in len_graphemes.items():
            nest.Connect(grapheme_col,
                         sum([list(neurs)
                              for hypercol in grapheme_hypercolumns
                              for (label, neurs) in hypercol.items()
                              if label == grapheme], []),
                         syn_spec='head_grapheme_synapse_model')
    for (word, word_col) in lexical_cols.items():
        word_decomposition = decompose_word(word)
        for (hcol_n, hypercol) in enumerate(grapheme_hypercolumns):
            if hcol_n == len(word_decomposition):
                break
            nest.Connect(word_col, hypercol[word_decomposition[hcol_n]],
                         syn_spec=prm['lexical_grapheme_excitation'])

    # Insert probes:
    for (word, word_col) in lexical_cols.items():
        insert_probe(word_col, word)
    insert_probe(lexical_inhibiting_population, 'lexical_inhibition')
    ##for (letter, letter_col) in letter_hypercolumns[1].items():
    ##    insert_probe(letter_col, 'L2-'+letter)
    for len_graphemes in reading_head:
        for (grapheme, grapheme_col) in len_graphemes.items():
            insert_probe(grapheme_col, 'head-'+grapheme)
    # [Reading facility config:]
    spike_groups['Head'] = ['head-'+g for g in prm['graphemes']]
    spike_groups['Words'] = prm['vocabulary']
    spike_decisions['Reading'] = []
    for (hcol_n, hypercol) in enumerate(grapheme_hypercolumns):
        spike_decisions['Reading'].append([])
        for (grapheme, grapheme_col) in hypercol.items():
            insert_probe(grapheme_col, 'g{}-{}'.format(hcol_n, grapheme))
            spike_decisions['Reading'][-1].append('g{}-{}'.format(hcol_n, grapheme))

    # Run the simulation, write readings.
    nest.Simulate(prm['letter_focus_time'])
    for lett_n in range(prm['max_text_len']):
        # Reassign the letter -> head weights.
        weights_dist = stats.skewnorm(6, loc=lett_n-0.7, scale=0.67)
        for assg_lett_n in range(prm['max_text_len']):
            assg_hypercol = sum([list(col) for (lett, col) in letter_hypercolumns[assg_lett_n].items()], [])
            for (ln, len_graphemes) in enumerate(reading_head):
                len_graphemes = sum([list(col) for (lett, col) in len_graphemes.items()], [])
                if len(len_graphemes) == 0:
                    continue
                nest.SetStatus( nest.GetConnections(assg_hypercol, len_graphemes),
                                { 'weight' : (  weights_dist.pdf(assg_lett_n) * 3000
                                              / (1.0 + (ln-1)*prm['grapheme_length_damping'])) })
        nest.Simulate(prm['letter_focus_time'])

def word_read():
    try:
        nest.GetStatus((1,))
    except nest.pynestkernel.NESTError:
        raise RuntimeError('calling word_read with no simulation state available')

    word_decisions = decide_spikes(spike_decisions['Reading'])
    stop_boundary = prm['max_text_len']
    for dec_n in range(1, len(word_decisions)):
        if word_decisions[dec_n][1] < word_decisions[dec_n-1][1] * 0.5:
            stop_boundary = dec_n
            break

    return ''.join([dec[0][dec[0].index('-')+1:] for dec in word_decisions[:stop_boundary]])

def save_readings(simulation_name):
    write_readings(prm['readings_path']+simulation_name,
                   params=prm,
                   spike_groups=spike_groups,
                   spike_decisions=spike_decisions)