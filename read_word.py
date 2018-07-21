import sys # import the model/nest later, so it won't display its startup message if we fail on config

# Parse command line args.
if not (len(sys.argv) in [2, 3]):
    print('Usage: read_word.py TEXT_INPUT [EXPERIMENT_NAME]')
    sys.exit(-1)
net_text_input = sys.argv[1]

experiment_name = 'experim'
if len(sys.argv) == 3:
    experiment_name = sys.argv[2]
word_simulation_name = experiment_name + '_' + net_text_input

from reading_model import simulate_reading, save_readings

simulate_reading(net_text_input)
save_readings(word_simulation_name)
