import datetime, os, sys, nest, pylab

# NOTE good values of record_from depend on neuron type used; there need be code for ploting for each below
multimeter_params = { 'withtime': True, 'record_from': ['V_m'] }
spikedet_params = { 'withgid': True, 'withtime': True }

probes = dict()

def reset_reporting():
    probes.clear()

def insert_probe(place, name):
    if name in probes:
        raise KeyError('probe name {} already in use'.format(name))
    probes[name] = { 'multimeter': nest.Create('multimeter', params=multimeter_params),
                     'spikedet': nest.Create('spike_detector', params=spikedet_params) }
    nest.Connect(probes[name]['multimeter'], place)
    nest.Connect(place, probes[name]['spikedet'])

def score_spikes(names):
    "Return a sorted list of (spike counts, names) for a list of names registered for reporting."
    contest_probes = [(len(nest.GetStatus(probe['spikedet'], 'events')[0]['times']), name)
                      for (name, probe) in probes.items()
                      if name in names]
    contest_probes.sort(key=lambda x: x[0], reverse=True)
    return contest_probes

def decide_spikes(name_groups):
    "Given a list of lists of names registered for reporting, for each list (group) return a list of pairs (name, spike count) with most spikes."
    decisions = []
    for group in name_groups:
        group_probes = score_spikes(group)
        decisions.append((group_probes[0][1], group_probes[0][0]))
    return decisions

def write_readings(path, params=None, spike_groups=dict(), spike_decisions=dict(), skip_charts=False):
    path += datetime.datetime.now().strftime('_%d-%m-%Y_%H-%M-%S')+'/' # add a timestamp
    os.makedirs(path, exist_ok=False) # throw an exception if exists
    if not skip_charts:
        for (name, probe) in probes.items():
            multim_events = nest.GetStatus(probe['multimeter'], 'events')[0]
            fig = pylab.figure()
            pylab.plot(multim_events['times'], multim_events['V_m'])
            pylab.ticklabel_format(useOffset=False, style='plain') # disable offsets and scientific notation
            pylab.savefig(path+name+'_membrane_potential.png')
            pylab.close(fig)

            spikedet_events = nest.GetStatus(probe['spikedet'], 'events')[0]
            fig = pylab.figure()
            pylab.plot(spikedet_events['times'], spikedet_events['senders'], '.')
            pylab.ticklabel_format(useOffset=False, style='plain')
            pylab.savefig(path+name+'_spikes.png')
            pylab.close(fig)

    for (label, names) in spike_groups.items():
        with open(path+label+'_spike_scores.txt', 'w+') as spike_group_file:
            contest_probes = score_spikes(names)
            print('# '+label, file=spike_group_file)
            for (spike_count, name) in contest_probes:
                print('{:>7} : {}'.format(spike_count, name), file=spike_group_file)

    for (label, groups) in spike_decisions.items():
        with open(path+label+'_spike_decision.txt', 'w+') as spike_decision_file:
            decisions = decide_spikes(groups)
            for dec in decisions:
                print(dec[1], dec[0], file=spike_decision_file)

    if params is not None:
        with open(path+'_params.txt', 'w+') as params_file:
            for (param, val) in params.items():
                print('{} : {}'.format(param, val), file=params_file)
