from reading_model import simulate_reading, word_read

cases = [
            ('else', 'else'),
            ('lease', 'lease'),
            ('least', 'least'),
            ('lute', 'lute'),
            ('sell', 'sell'),
            ('sells', 'sells'),
            ('set', 'set'),
            ('stu', 'stu'),
            ('tall', 'tall'),
            ('tell', 'tell'),
            ('tells', 'tells'),
            ('zet', 'zet'),
            ('zest', 'zest')
        ]
observations = []

for (word, expected_form) in cases:
    simulate_reading(word)
    prediction = word_read()
    observations.append((word, prediction))

print('=== Observations ===')
for (word, prediction) in observations:
    print(word, prediction)
