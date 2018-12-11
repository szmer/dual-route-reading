from itertools import chain
from scipy import stats
from unidecode import unidecode
from statistics import mean
from collections import ChainMap
import nest
from levenshtein import distance_within
from nltk.probability import FreqDist
from neuro_reporting import reset_reporting, insert_probe, write_readings, decide_spikes

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

def syn_config(param, *args):
    if not args:
        return ChainMap(param, prm['synapse_default'])
    return ChainMap(param(*args), prm['synapse_default'])

# Global config.
prm = {
        'letter_focus_time': 50.0,
        'decision_threshold': 150.0,
        'readings_path': 'readings/',

        'neuron_type': 'iaf_psc_alpha',
        'synapse_default': { 'model': 'stdp_synapse', 'alpha': 1.0 }, # default configuration for the network
        'letter_neuron_params_on': { 'I_e': 900.0 }, # constant input current in pA
        'letters_poisson_generator': { 'start': 0.0,
                                       'stop': 99999.0,
                                       'rate': 750.0 },
        'letter_column_size': 4,
        'head_column_size': 12,
        'grapheme_column_size': 8,
        'lexical_column_size': 1,
        'lexical_inhibiting_pop_size': 10,
        # Synapse specifications.
        'poisson_letter_excitation': { 'weight': 1000.0 },
        'letter_col_lateral_inhibition': { 'weight': -100.0 },
        'letter_head_excitation': { 'weight': 300.0 },
        'member_first_letter_excitation': { 'weight': 3500.0 },
        'member_last_letter_excitation': { 'weight': 190.0 },
        'member_letter_excitation_weight': 10000.0, # make them separate so they show up in readings printouts
        'absent_letter_inhibition_weight': -1140.0,
        'member_letter_excitation': (lambda length: { 'weight': prm['member_letter_excitation_weight'] / (length * 6) }),
        'absent_letter_inhibition': (lambda length: { 'weight': prm['absent_letter_inhibition_weight'] / length }),
        'member_letter_excitation_suffix': (lambda length: { 'weight': prm['member_letter_excitation_weight'] / (length*5) / 6 }),
        'absent_letter_inhibition_suffix': (lambda length: { 'weight': prm['absent_letter_inhibition_weight'] / (length*2) / 6 }),
        'shorter_word_inhibition': { 'weight': -40.0 },
        'lexical_grapheme_base_excitation_weight': 3600.0,
        'lexical_inhibiting_pop_excitation': { 'weight': 9500.0 }, # this makes the strongest lexical matches relatively stronger
        'lexical_inhibiting_pop_feedback_weight': -900.0,
        'lexical_inhibiting_pop_feedback': lambda length: { 'weight': length*prm['lexical_inhibiting_pop_feedback_weight'] },
        'lexical_lateral_inhibition': { 'weight': -50.0 }, # of similar words
        'suffix_lateral_inhibition': { 'weight': -1100.0 }, # of all other suffixes
        'suffix_grapheme_base_weight': 4000.0, # parametrized by distance from the estimated stem end
        'grapheme_lateral_inhibition_weight': -30.0,
        'grapheme_lateral_inhibition': (lambda length: { 'weight': prm['grapheme_lateral_inhibition_weight'] * length }),
        # weights letter -> head are divided by (1 + (target_grapheme_len-1)*this)
        'grapheme_length_damping': 0.8,
        'grapheme_lexical_feedback': { 'weight': 5.0 },
        'head_grapheme_base_weight': 4500.0,
        'head_grapheme_synapse_model': { 'U': 0.67, 'u': 0.67, 'x': 1.0, 'tau_rec': 50.0,
                                        'tau_fac': 0.0 },
        'letter_lexical_synapse_model': { 'U': 0.67, 'u': 0.67, 'x': 1.0, 'tau_rec': 100.0,
                                        'tau_fac': 1500.0 },
        # (the _model part in name is meant to mark that we register a separate synampse 'type')
        }

# These have to be declared globally to be available to separate saving functions.
spike_groups, spike_decisions = {}, {} # to be filled when preparing a simulation

###    local_vocabulary = [w for w in vocabulary[unidecode(net_text_input[0])]
###                        if distance_within(w, net_text_input, 4)] # NOTE we may compare only stems to the full input!
class DrcNetwork():
    def __init__(self, letters, graphemes, local_vocabulary, suffixes=False, num_threads=9, max_text_len=10)
        """Build the network."""
        self.graphemes_by_lengths = [[g for g in graphemes if len(g) == l]
                                for l in range(max([len(g) for g in graphemes])+1)]
        self.letters, self.graphemes, self.local_vocabulary, self.suffixes = letters, graphemes, local_vocabulary, suffixes
        self.max_text_len = max_text_len
        self.prm = prm # for easier inspection from the outside of this file

        nest.ResetKernel()
        nest.SetKernelStatus({'local_num_threads': num_threads})

        nest.CopyModel('tsodyks2_synapse', 'head_grapheme_synapse_model', prm['head_grapheme_synapse_model'])
        nest.CopyModel('tsodyks2_synapse', 'letter_lexical_synapse_model', prm['letter_lexical_synapse_model'])

        self.graphemes_dist = FreqDist(chain.from_iterable([decompose_word(w) for w in self.local_vocabulary]))

        self.lexical_cols = dict([(w, nest.Create(prm['neuron_type'], prm['lexical_column_size']))
                             for w in self.local_vocabulary])
        if self.suffixes:
            self.suffixes_cols = dict([(s, nest.Create(prm['neuron_type'], prm['lexical_column_size']))
                                for s in self.suffixes])
        self.lexical_inhibiting_population = nest.Create(prm['neuron_type'], prm['lexical_inhibiting_pop_size'])
        self.letter_hypercolumns = [make_hypercolumn(self.letters, prm['letter_column_size'])
                               for i in range(self.max_text_len)]
        # Reading heads' columns are sorted in separate lists by grapheme lengths.
        self.reading_head_len_sorted = [make_hypercolumn(size_graphemes, prm['head_column_size'])
                                  for size_graphemes in self.graphemes_by_lengths]
        self.reading_head = {} # a 'flat' version
        for len_graphemes in self.reading_head_len_sorted:
            self.reading_head.update(len_graphemes)
        self.grapheme_hypercolumns = [make_hypercolumn(self.graphemes, prm['grapheme_column_size'])
                                 for i in range(self.max_text_len)]

        # Make connections.
        ###self.end_weight_dist = stats.norm(loc=len(net_text_input), scale=3.0) # for exciting suffixes
        for (hcol_n, hypercol) in enumerate(self.letter_hypercolumns): # hypercol is: letter -> (neuron's nest id)
            # Turn on appropriate letter columns.
            if hcol_n < len(net_text_input) and net_text_input[hcol_n] in self.letters:
                poisson_gen = nest.Create('poisson_generator', 1, prm['letters_poisson_generator'])
                nest.Connect(poisson_gen, hypercol[net_text_input[hcol_n]], syn_spec=syn_config(prm['poisson_letter_excitation']))
                ###nest.SetStatus(hypercol[net_text_input[hcol_n]], prm['letter_neuron_params_on'])

            # Letter hypercol's lateral inhibition to subsequent hypercols
            for hypercol2 in self.letter_hypercolumns[hcol_n+1:]:
                nest.Connect(all_columns_cells(hypercol), all_columns_cells(hypercol2),
                             syn_spec=syn_config(prm['letter_col_lateral_inhibition']))
            # Letter hypercol -> the reading head
            for (letter, letter_col) in hypercol.items():
                for (grapheme, grapheme_col) in self.reading_head.items():
                    if letter in grapheme:
                        nest.Connect(letter_col, grapheme_col, syn_spec=syn_config(prm['letter_head_excitation']))
            # Letter hypercol -> lexical units
            for (word, word_col) in self.lexical_cols.items():
                if hcol_n >= len(word):
                    nest.Connect(all_columns_cells(hypercol), word_col, syn_spec=syn_config(prm['shorter_word_inhibition']))
                else:
                    for (letter, letter_col) in hypercol.items():
                        if hcol_n == 0 and unidecode(word[hcol_n]) == unidecode(letter):
                            nest.Connect(letter_col, word_col, syn_spec=syn_config('letter_lexical_synapse_model'))
                            nest.SetStatus(nest.GetConnections(letter_col, word_col),
                                           prm['member_first_letter_excitation'])
                        if (not self.suffixes
                                and hcol_n == len(word)-1 and unidecode(word[len(word)-1]) == unidecode(letter)):
                            nest.Connect(letter_col, word_col, syn_spec=syn_config('letter_lexical_synapse_model'))
                            nest.SetStatus(nest.GetConnections(letter_col, word_col),
                                           prm['member_last_letter_excitation'])
                        elif unidecode(letter) in unidecode(word):
                            nest.Connect(letter_col, word_col, syn_spec=syn_config('letter_lexical_synapse_model'))
                            nest.SetStatus(nest.GetConnections(letter_col, word_col),
                                           prm['member_letter_excitation'](len(word)))
                        else:
                            nest.Connect(letter_col, word_col, syn_spec=syn_config('letter_lexical_synapse_model'))
                            nest.SetStatus(nest.GetConnections(letter_col, word_col),
                                           prm['absent_letter_inhibition'](len(word)))
            # Letter hypercol -> suffixes units
            if self.suffixes:
                for (suffix, suffix_col) in self.suffixes_cols.items():
                    if len(net_text_input)-hcol_n <= len(suffix):
                        for (letter, letter_col) in hypercol.items():
                            if letter in suffix:
                                nest.Connect(letter_col, suffix_col,
                                            syn_spec=syn_config(prm['member_letter_excitation_suffix'](len(suffix))))
                            else:
                                nest.Connect(letter_col, suffix_col,
                                            syn_spec=syn_config(prm['absent_letter_inhibition_suffix'](len(suffix))))
        for (grapheme, grapheme_col) in self.reading_head.items():
            nest.Connect(grapheme_col,
                         sum([list(neurs)
                              for hypercol in self.grapheme_hypercolumns
                              for (label, neurs) in hypercol.items()
                              if label == grapheme], []),
                         syn_spec=syn_config('head_grapheme_synapse_model'))
        nest.Connect(all_columns_cells(self.lexical_cols), self.lexical_inhibiting_population,
                     syn_spec=syn_config(prm['lexical_inhibiting_pop_excitation']))
        for (word, word_col) in self.lexical_cols.items():
            nest.Connect(self.lexical_inhibiting_population, word_col,
                        syn_spec=syn_config(prm['lexical_inhibiting_pop_feedback'](len(word))))
            word_decomposition = decompose_word(word)
            for (hcol_n, hypercol) in enumerate(self.grapheme_hypercolumns):
                if hcol_n == len(word_decomposition):
                    break
                nest.Connect(word_col, hypercol[word_decomposition[hcol_n]],
                             syn_spec=syn_config({ 'weight': (prm['lexical_grapheme_base_excitation_weight']
                                                    # the excitation is stronger with rarer letters:
                                                    / self.graphemes_dist.freq(word_decomposition[hcol_n]))}))
                # Grapheme -> lexical feedback.
                nest.Connect(hypercol[word_decomposition[hcol_n]], word_col,
                             syn_spec=syn_config(prm['grapheme_lexical_feedback']))
            # Lateral inhibition for similar words.
            for (word2, word2_col) in self.lexical_cols.items():
                if word2 == word:
                    continue
                elif distance_within(word, word2, 4):
                    nest.Connect(word_col, word2_col, syn_spec=syn_config(prm['lexical_lateral_inhibition']))
        if self.suffixes:
            for (suffix, suffix_col) in self.suffixes_cols.items():
                # Lateral inhibition for suffixes.
                for (suffix2, suffix2_col) in  self.suffixes_cols.items():
                    if suffix != suffix2:
                        nest.Connect(suffix_col, suffix2_col, syn_spec=syn_config(prm['suffix_lateral_inhibition']))
                # Suffix -> grapheme connections.
                for (hcol_n, hypercol) in enumerate(self.grapheme_hypercolumns):
                    # (weights will be assigned dynamically later)
                    nest.Connect(suffix_col, all_columns_cells(hypercol), syn_spec=syn_config({ 'weight': 0.0 }))
        for (hcol_n, hypercol) in enumerate(self.grapheme_hypercolumns):
            for (grapheme, col) in hypercol.items():
                # Lateral inhibition of graphemes containing at least one same letter
                if hcol_n != 0:
                    for similar_grapheme in [g for g in self.graphemes if len(set(g).union(set(grapheme))) > 0]:
                        nest.Connect(col, self.grapheme_hypercolumns[hcol_n-1][similar_grapheme],
                                     syn_spec=syn_config(prm['grapheme_lateral_inhibition'](len(similar_grapheme))))
                if hcol_n+1 != self.max_text_len:
                    for similar_grapheme in [g for g in self.graphemes if len(set(g).union(set(grapheme))) > 0]:
                        nest.Connect(col, self.grapheme_hypercolumns[hcol_n+1][similar_grapheme],
                                     syn_spec=syn_config(prm['grapheme_lateral_inhibition'](len(similar_grapheme))))

   def setup_reporting(self):
        # Re-setup reporting.
        reset_reporting()
        spike_groups.clear()
        spike_decisions.clear()
        # Insert probes:
        for (word, word_col) in self.lexical_cols.items():
            insert_probe(word_col, word, always_chart=False)
        if self.suffixes:
            for (suffix, suffix_col) in self.suffixes_cols.items():
                insert_probe(suffix_col, 'suff_'+suffix, always_chart=False)
        insert_probe(self.lexical_inhibiting_population, 'lexical_inhibition')
        ##for (letter, letter_col) in self.letter_hypercolumns[1].items():
        ##    insert_probe(letter_col, 'L2-'+letter)
        for (grapheme, grapheme_col) in self.reading_head.items():
            insert_probe(grapheme_col, 'head-'+grapheme, always_chart=False)

        # [Reading facility config:]
        spike_groups['Head'] = ['head-'+g for g in self.graphemes]
        spike_groups['Words'] = self.local_vocabulary
        if self.suffixes:
            spike_groups['Suffixes'] = ['suff_'+suff for suff in self.suffixes]
            spike_decisions['Stems'] = [ self.local_vocabulary ]
        spike_decisions['Reading'] = []
        for (hcol_n, hypercol) in enumerate(self.grapheme_hypercolumns):
            spike_decisions['Reading'].append([])
            for (grapheme, grapheme_col) in hypercol.items():
                insert_probe(grapheme_col, 'g{}-{}'.format(hcol_n, grapheme), always_chart=False)
                spike_decisions['Reading'][-1].append('g{}-{}'.format(hcol_n, grapheme))

    def simulate_reading(self, net_text_input):
       """Returns the time when the reading simulation ended for this input."""
        if len(net_text_input) > self.max_text_len:
            raise ValueError('Text input {} has to be shorter than max_text_len: {}'.format(net_text_input, self.max_text_len))

        start_time = nest.GetKernelStatus('time')

        nest.Simulate(prm['letter_focus_time'])
        for step_n in range(self.max_text_len):

            # Reassign the letter -> head weights (shifting skew normal).
            weights_dist = stats.skewnorm(6, loc=step_n-0.7, scale=0.67)
            for assg_lett_n in range(self.max_text_len):
                assg_hypercol = all_columns_cells(self.letter_hypercolumns[assg_lett_n])
                for (ln, len_graphemes) in enumerate(self.reading_head_len_sorted):
                    len_graphemes = all_columns_cells(len_graphemes)
                    if len(len_graphemes) == 0:
                        continue
                    nest.SetStatus( nest.GetConnections(assg_hypercol, len_graphemes),
                                    { 'weight' : (  weights_dist.pdf(assg_lett_n) * 3000
                                                  / (1.0 + (ln-1)*prm['grapheme_length_damping'])) })

            # Reassign the head -> grapheme weights (normal parametrized by time for each target hypercolumn).
            for (hcol_n, hypercol) in enumerate(self.grapheme_hypercolumns):
                weights_dist = stats.norm(loc=hcol_n+1, scale=1.0) # hcol_n is treated as time step number
                                                                   # (add one because of the first "dummy" step)
                nest.SetStatus( nest.GetConnections(all_columns_cells(self.reading_head),
                                                    all_columns_cells(hypercol)),
                                { 'weight' : weights_dist.pdf(1.0 + (nest.GetKernelStatus('time')-start_time) / prm['letter_focus_time'])
                                             * prm['head_grapheme_base_weight'] })

            # Reassign the suffix -> grapheme weights (depending on estimated stem end).
            if self.suffixes:#### and step_n > len(net_text_input)/2:
                stem_end = mean([len(stem_reading[0]) for stem_reading in decide_spikes(spike_decisions['Stems'])[:15]])
                #print(stem_end)
                for (suffix, suffix_col) in self.suffixes_cols.items():
                    suffix_decomposition = decompose_word(suffix)
                    for grapheme in set(suffix_decomposition):
                        # Each occurence of a grapheme in suffix must exert is
                        # influence individually, they are then summed.
                        indices = [gi for (gi, g) in enumerate(suffix_decomposition) if g == grapheme]
                        weight_dists = [stats.norm(loc=stem_end+ind, scale=3.0)
                                         for ind in indices]
                        if len(weight_dists) == 0:
                            continue
                        for (hcol_n, hypercol) in enumerate(self.grapheme_hypercolumns):
                            #print('stem_end', stem_end, 'hcol', hcol_n, weight_dists[0].pdf(hcol_n))
                            nest.SetStatus( nest.GetConnections(suffix_col, hypercol[grapheme]),
                                            { 'weight': sum( [dist.pdf(hcol_n) for dist in weight_dists] )
                                                        * prm['suffix_grapheme_base_weight'] })

            nest.Simulate(prm['letter_focus_time'])

    return start_time, nest.GetKernelStatus('time')

    # NOTE this needs to be kept up to date with the simulation method
    def word_simulation_time(self):
        return (1+self.max_text_len)*prm['letter_focus_time']

    def word_read(self, end_time):
        if nest.GetKernelStatus('time') < end_time:
            raise RuntimeError('calling word_read with no simulation state available for the given end time')

        word_decisions = decide_spikes(spike_decisions['Reading'], # get columns with their spike counts
                                       end_time-self.word_simulation_time, end_time)
        stop_boundary = self.max_text_len
        for dec_n in range(1, len(word_decisions)):
            if word_decisions[dec_n][1] < prm['decision_threshold']:#word_decisions[dec_n-1][1] * 0.56:
                stop_boundary = dec_n
                break

        return ''.join([dec[0][dec[0].index('-')+1:] for dec in word_decisions[:stop_boundary]])

    def save_readings(self, simulation_name, start_time, end_time, thorough=True):
        write_readings(prm['readings_path']+simulation_name,
                       start_time, end_time,
                       params=prm,
                       spike_groups=spike_groups,
                       spike_decisions=spike_decisions,
                       thorough=thorough)
