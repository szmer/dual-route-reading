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
            ('zest', 'zest'),
            ('lese', 'lease'),
            ('ltte', 'lute'),
            ('elsee', 'else'),
            ('sett', 'set'),
            ('tal', 'tall'),
            ('sels', 'sells'),
            ('ltue', 'lute'),
            ('tlal', 'tall'),
            ('laese', 'lease')
        ]
observations = []

good = 0
for (word, expected_form) in cases:
    simulate_reading(word)
    prediction = word_read()
    grade = 'x'
    if prediction == expected_form:
        good += 1
        grade = ''
    observations.append((word, prediction, grade))

print('=== Observations ===')
for (word, prediction, grade) in observations:
    print(word, prediction, grade)
print('accuracy', good / len(cases))
