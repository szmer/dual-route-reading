from scipy import stats
from unidecode import unidecode
from statistics import mean
import nest
from neuro_reporting import reset_reporting, insert_probe, write_readings, decide_spikes
from levenshtein import distance_within

###nest.set_verbosity('M_ERROR') # don't print detailed simulation info

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

def all_columns_cells(hypercol):
    return sum([list(col) for (label, col) in hypercol.items()], [])

# Global config.
prm = {
        'max_text_len': 12,
        'letter_focus_time': 50.0,
        'decision_threshold': 100.0,
        'readings_path': 'readings/',
        'language_data_path': './pol/',
        'stems_and_suffixes': True,

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
        'member_last_letter_excitation': { 'weight': 190.0 },
        'member_letter_excitation_weight': 540.0, # make them separate so they show up in readings printouts
        'absent_letter_inhibition_weight': -160.0,
        'member_letter_excitation': (lambda length: { 'weight': prm['member_letter_excitation_weight'] / (length*38) }),
        'absent_letter_inhibition': (lambda length: { 'weight': prm['absent_letter_inhibition_weight'] / (length*6) }),
        'member_letter_excitation_suffix': (lambda length: { 'weight': 10*prm['member_letter_excitation_weight'] / (length*10) }),
        'absent_letter_inhibition_suffix': (lambda length: { 'weight': 10*prm['absent_letter_inhibition_weight'] / (length*4) }),
        'shorter_word_inhibition': { 'weight': -40.0 },
        'lexical_grapheme_excitation': { 'weight': 1000.0 },
        'lexical_inhibiting_pop_excitation': { 'weight': 650.0 }, # this makes the strongest lexical matches relatively stronger
        'lexical_inhibiting_pop_feedback': { 'weight': -300.0 },
        'lexical_lateral_inhibition': { 'weight': -50.0 }, # of similar words
        'suffix_lateral_inhibition': { 'weight': -1100.0 }, # of all other suffixes
        'letter_suffix_excitation': { 'weight': 600.0 },
        'suffix_grapheme_base_weight': 24000.0, # parametrized by distance from the estimated stem end
        'grapheme_lateral_inhibition_weight': -25.0,
        'grapheme_lateral_inhibition': (lambda length: { 'weight': prm['grapheme_lateral_inhibition_weight'] * length }),
        # weights letter -> head are divided by (1 + (target_grapheme_len-1)*this)
        'grapheme_length_damping': 0.8,
        'grapheme_lexical_feedback': { 'weight': 50.0 },
        'head_grapheme_base_weight': 3000.0,
        'head_grapheme_synapse_model': { 'U': 0.67, 'u': 0.67, 'x': 1.0, 'tau_rec': 50.0,
                                        'tau_fac': 0.0 },
        # (the _model part in name is meant to mark that we register a separate synampse 'type')
        }

# Read language data.
letters, graphemes, vocabulary = None, None, dict()
with open(prm['language_data_path']+'letters') as fl:
    letters = fl.read().strip().split()
with open(prm['language_data_path']+'graphemes') as fl:
    graphemes = fl.read().strip().split()
# Vocabulary is sorted by the first letter.
skipped_words_n = 0
vocabulary_path = prm['language_data_path'] + ('stems'
                                               if prm['stems_and_suffixes']
                                               else 'vocabulary')
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
if prm['stems_and_suffixes']:
    with open(prm['language_data_path'] + 'suffixes') as fl:
        for line in fl:
            suffix = line.strip()
            suffixes.append(suffix)

graphemes_by_lengths = [[g for g in graphemes if len(g) == l]
                        for l in range(max([len(g) for g in graphemes])+1)]

# These have to be declared globally to be available to separate saving functions.
spike_groups, spike_decisions = {}, {} # to be filled when preparing a simulation

def simulate_reading(net_text_input):
    if len(net_text_input) > prm['max_text_len']:
        raise ValueError('Text input {} has to be shorter than max_text_len: {}'.format(net_text_input, prm['max_text_len']))

    # Build the network.
    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': 9})
    reset_reporting()
    spike_groups.clear()
    spike_decisions.clear()

    nest.CopyModel('tsodyks2_synapse', 'head_grapheme_synapse_model', prm['head_grapheme_synapse_model'])

    local_vocabulary = [w for w in vocabulary[unidecode(net_text_input[0])]
                        if distance_within(w, net_text_input, 4)]

    lexical_cols = dict([(w, nest.Create(prm['neuron_type'], prm['lexical_column_size']))
                         for w in local_vocabulary])
    if prm['stems_and_suffixes']:
        suffixes_cols = dict([(s, nest.Create(prm['neuron_type'], prm['lexical_column_size']))
                            for s in suffixes])
    lexical_inhibiting_population = nest.Create(prm['neuron_type'], prm['lexical_inhibiting_pop_size'])
    letter_hypercolumns = [make_hypercolumn(letters, prm['letter_column_size'])
                           for i in range(prm['max_text_len'])]
    # Reading heads' columns are sorted in separate lists by grapheme lengths.
    reading_head_len_sorted = [make_hypercolumn(size_graphemes, prm['head_column_size'])
                              for size_graphemes in graphemes_by_lengths]
    reading_head = {} # a 'flat' version
    for len_graphemes in reading_head_len_sorted:
        reading_head.update(len_graphemes)
    grapheme_hypercolumns = [make_hypercolumn(graphemes, prm['grapheme_column_size'])
                             for i in range(prm['max_text_len'])]

    # Make connections.
    end_weight_dist = stats.norm(loc=len(net_text_input), scale=3.0) # for exciting suffixes
    for (hcol_n, hypercol) in enumerate(letter_hypercolumns): # hypercol is: letter -> (neuron's nest id)
        # Turn on appropriate letter columns.
        if hcol_n < len(net_text_input) and net_text_input[hcol_n] in letters:
            nest.SetStatus(hypercol[net_text_input[hcol_n]], prm['letter_neuron_params_on'])

        # Letter hypercol's lateral inhibition to subsequent hypercols
        for hypercol2 in letter_hypercolumns[hcol_n+1:]:
            nest.Connect(all_columns_cells(hypercol), all_columns_cells(hypercol2),
                         syn_spec=prm['letter_col_lateral_inhibition'])
        # Letter hypercol -> the reading head
        for (letter, letter_col) in hypercol.items():
            for (grapheme, grapheme_col) in reading_head.items():
                if letter in grapheme:
                    nest.Connect(letter_col, grapheme_col, syn_spec=prm['letter_head_excitation'])
        # Letter hypercol -> lexical units
        for (word, word_col) in lexical_cols.items():
            if hcol_n >= len(word):
                nest.Connect(all_columns_cells(hypercol), word_col, syn_spec=prm['shorter_word_inhibition'])
            else:
                for (letter, letter_col) in hypercol.items():
                    if hcol_n == 0 and unidecode(word[hcol_n]) == unidecode(letter):
                        nest.Connect(letter_col, word_col, syn_spec=prm['member_first_letter_excitation'])
                    if (not prm['stems_and_suffixes']
                            and hcol_n == len(word)-1 and unidecode(word[len(word)-1]) == unidecode(letter)):
                        nest.Connect(letter_col, word_col, syn_spec=prm['member_last_letter_excitation'])
                    elif unidecode(letter) in unidecode(word):
                        nest.Connect(letter_col, word_col, syn_spec=prm['member_letter_excitation'](len(word)))
                    else:
                        nest.Connect(letter_col, word_col, syn_spec=prm['absent_letter_inhibition'](len(word)))
        # Letter hypercol -> suffixes units
        if prm['stems_and_suffixes']:
            for (suffix, suffix_col) in suffixes_cols.items():
                if len(net_text_input)-hcol_n <= len(suffix):
                    for (letter, letter_col) in hypercol.items():
                        if letter in suffix:
                            nest.Connect(letter_col, suffix_col,
                                        syn_spec=prm['member_letter_excitation_suffix'](len(suffix)))
                        else:
                            nest.Connect(letter_col, suffix_col,
                                        syn_spec=prm['absent_letter_inhibition_suffix'](len(suffix)))
    nest.Connect(all_columns_cells(lexical_cols), lexical_inhibiting_population,
                 syn_spec=prm['lexical_inhibiting_pop_excitation'])
    nest.Connect(lexical_inhibiting_population, all_columns_cells(lexical_cols),
                 syn_spec=prm['lexical_inhibiting_pop_feedback'])
    for (grapheme, grapheme_col) in reading_head.items():
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
            # Grapheme -> lexical feedback.
            nest.Connect(hypercol[word_decomposition[hcol_n]], word_col,
                         syn_spec=prm['grapheme_lexical_feedback'])
        # Lateral inhibition for similar words.
        for (word2, word2_col) in lexical_cols.items():
            if word2 == word:
                continue
            if distance_within(word, word2, 2):
                nest.Connect(word_col, word2_col, syn_spec=prm['lexical_lateral_inhibition'])
    if prm['stems_and_suffixes']:
        for (suffix, suffix_col) in suffixes_cols.items():
            # Lateral inhibition for suffixes.
            for (suffix2, suffix2_col) in  suffixes_cols.items():
                if suffix != suffix2:
                    nest.Connect(suffix_col, suffix2_col, syn_spec=prm['suffix_lateral_inhibition'])
            # Suffix -> grapheme connections.
            for (hcol_n, hypercol) in enumerate(grapheme_hypercolumns):
                # (weights will be assigned dynamically later)
                nest.Connect(suffix_col, all_columns_cells(hypercol), syn_spec={ 'weight': 0.0 })
    for (hcol_n, hypercol) in enumerate(grapheme_hypercolumns):
        for (grapheme, col) in hypercol.items():
            # Lateral inhibition of graphemes containing at least one same letter
            if hcol_n != 0:
                for similar_grapheme in [g for g in graphemes if len(set(g).union(set(grapheme))) > 0]:
                    nest.Connect(col, grapheme_hypercolumns[hcol_n-1][similar_grapheme],
                                 syn_spec=prm['grapheme_lateral_inhibition'](len(similar_grapheme)))
            if hcol_n+1 != prm['max_text_len']:
                for similar_grapheme in [g for g in graphemes if len(set(g).union(set(grapheme))) > 0]:
                    nest.Connect(col, grapheme_hypercolumns[hcol_n+1][similar_grapheme],
                                 syn_spec=prm['grapheme_lateral_inhibition'](len(similar_grapheme)))

    # Insert probes:
    for (word, word_col) in lexical_cols.items():
        insert_probe(word_col, word)
    if prm['stems_and_suffixes']:
        for (suffix, suffix_col) in suffixes_cols.items():
            insert_probe(suffix_col, suffix)
    insert_probe(lexical_inhibiting_population, 'lexical_inhibition')
    ##for (letter, letter_col) in letter_hypercolumns[1].items():
    ##    insert_probe(letter_col, 'L2-'+letter)
    for (grapheme, grapheme_col) in reading_head.items():
        insert_probe(grapheme_col, 'head-'+grapheme)
    # [Reading facility config:]
    spike_groups['Head'] = ['head-'+g for g in graphemes]
    spike_groups['Words'] = local_vocabulary
    if prm['stems_and_suffixes']:
        spike_groups['Suffixes'] = suffixes
        spike_decisions['Stems'] = [ local_vocabulary ]
    spike_decisions['Reading'] = []
    for (hcol_n, hypercol) in enumerate(grapheme_hypercolumns):
        spike_decisions['Reading'].append([])
        for (grapheme, grapheme_col) in hypercol.items():
            insert_probe(grapheme_col, 'g{}-{}'.format(hcol_n, grapheme))
            spike_decisions['Reading'][-1].append('g{}-{}'.format(hcol_n, grapheme))

    # Run the simulation, write readings.
    nest.Simulate(prm['letter_focus_time'])
    for step_n in range(prm['max_text_len']):

        # Reassign the letter -> head weights (shifting skew normal).
        weights_dist = stats.skewnorm(6, loc=step_n-0.7, scale=0.67)
        for assg_lett_n in range(prm['max_text_len']):
            assg_hypercol = all_columns_cells(letter_hypercolumns[assg_lett_n])
            for (ln, len_graphemes) in enumerate(reading_head_len_sorted):
                len_graphemes = all_columns_cells(len_graphemes)
                if len(len_graphemes) == 0:
                    continue
                nest.SetStatus( nest.GetConnections(assg_hypercol, len_graphemes),
                                { 'weight' : (  weights_dist.pdf(assg_lett_n) * 3000
                                              / (1.0 + (ln-1)*prm['grapheme_length_damping'])) })

        # Reassign the head -> grapheme weights (normal parametrized by time for each target hypercolumn).
        for (hcol_n, hypercol) in enumerate(grapheme_hypercolumns):
            weights_dist = stats.norm(loc=hcol_n+1, scale=1.0) # hcol_n is treated as time step number
                                                               # (add one because of the first "dummy" step)
            nest.SetStatus( nest.GetConnections(all_columns_cells(reading_head),
                                                all_columns_cells(hypercol)),
                            { 'weight' : weights_dist.pdf(nest.GetKernelStatus('time') / prm['letter_focus_time'])
                                         * prm['head_grapheme_base_weight'] })

        # Reassign the suffix -> grapheme weights (depending on estimated stem end).
        if prm['stems_and_suffixes']:#### and step_n > len(net_text_input)/2:
            stem_end = mean([len(stem_reading[0]) for stem_reading in decide_spikes(spike_decisions['Stems'])[:15]])
            for (suffix, suffix_col) in suffixes_cols.items():
                suffix_decomposition = decompose_word(suffix)
                for grapheme in set(suffix_decomposition):
                    # Each occurence of a grapheme in suffix must exert is
                    # influence individually, they are then summed.
                    indices = [gi for (gi, g) in enumerate(suffix_decomposition) if g == grapheme]
                    weight_dists = [stats.norm(loc=stem_end+ind, scale=3.0)
                                     for ind in indices]
                    if len(weight_dists) == 0:
                        continue
                    for (hcol_n, hypercol) in enumerate(grapheme_hypercolumns):
                        #print('stem_end', stem_end, 'hcol', hcol_n, weight_dists[0].pdf(hcol_n))
                        nest.SetStatus( nest.GetConnections(suffix_col, hypercol[grapheme]),
                                        { 'weight': sum( [dist.pdf(hcol_n) for dist in weight_dists] )
                                                    * prm['suffix_grapheme_base_weight'] })

        nest.Simulate(prm['letter_focus_time'])

def word_read():
    if nest.GetKernelStatus('time') == 0.0:
        raise RuntimeError('calling word_read with no simulation state available')

    word_decisions = decide_spikes(spike_decisions['Reading']) # get columns with their spike counts
    stop_boundary = prm['max_text_len']
    for dec_n in range(1, len(word_decisions)):
        if word_decisions[dec_n][1] < prm['decision_threshold']:#word_decisions[dec_n-1][1] * 0.56:
            stop_boundary = dec_n
            break

    return ''.join([dec[0][dec[0].index('-')+1:] for dec in word_decisions[:stop_boundary]])

def save_readings(simulation_name, skip_charts=False):
    write_readings(prm['readings_path']+simulation_name,
                   params=prm,
                   spike_groups=spike_groups,
                   spike_decisions=spike_decisions,
                   skip_charts=skip_charts)
