import csv, sys

if len(sys.argv) != 2:
    print('Usage: python3 word_list_test.py FILE_WITH_CASES')
    sys.exit(-1)

from reading_model import simulate_reading, word_read

cases = []
with open(sys.argv[1]) as fl:
    reader = csv.reader(fl)
    cases = [row[:2] for row in reader if row]

observations = []

good = 0
for (word, expected_form) in cases:
    try:
        simulate_reading(word)
    except ValueError:
        observations.append((word, '____', 'x'))
        continue
    prediction = word_read()
    grade = 'x'
    if prediction == expected_form:
        good += 1
        grade = ''
    observations.append((word, prediction, grade))
    print('.', end='')

print()
print('=== Observations ===')
for (word, prediction, grade) in observations:
    print(word, prediction, grade)
print('accuracy', good / len(cases))
