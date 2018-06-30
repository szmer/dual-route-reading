import datetime, os, sys, nest, pylab

# NOTE good values of record_from depend on neuron type used; there need be code for ploting for each below
multimeter_params = { 'withtime': True, 'record_from': ['V_m'] }
spikedet_params = { 'withgid': True, 'withtime': True }

probes = dict()

def insert_probe(place, name):
    if name in probes:
        raise KeyError('probe name {} already in use'.format(name))
    probes[name] = { 'multimeter': nest.Create('multimeter', params=multimeter_params),
                     'spikedet': nest.Create('spike_detector', params=spikedet_params) }
    nest.Connect(probes[name]['multimeter'], place)
    nest.Connect(place, probes[name]['spikedet'])

def print_spike_scores(names, file=sys.stdout):
    contest_probes = [(len(nest.GetStatus(probe['spikedet'], 'events')[0]['times']), name)
                      for (name, probe) in probes.items()
                      if name in names]
    contest_probes.sort(key=lambda x: x[0], reverse=True)
    for (spike_count, name) in contest_probes:
        print('{:>7} : {}'.format(spike_count, name), file=file)

def write_readings(path, spike_groups=dict()):
    path += datetime.datetime.now().strftime('_%d-%m-%Y_%H-%M-%S')+'/' # add a timestamp
    os.makedirs(path, exist_ok=False) # throw an exception if exists
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
            print_spike_scores(names, file=spike_group_file)
