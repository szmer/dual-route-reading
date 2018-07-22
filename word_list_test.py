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

good = 0
for (word, expected_form) in cases:
    simulate_reading(word)
    prediction = word_read()
    observations.append((word, prediction))
    if prediction == expected_form:
        good += 1

print('=== Observations ===')
for (word, prediction) in observations:
    print(word, prediction)
print('accuracy', good / len(cases))
