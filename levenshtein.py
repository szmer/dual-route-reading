# The MIT License (MIT)
# 
# Copyright (c) 2015 Jules Jacobs, (c) 2018 Szymon Rutkowski (modifications)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

class LevenshteinAutomaton:
    def __init__(self, string, n):
        self.string = string
        self.max_edits = n

    def start(self):
        return (range(self.max_edits+1), range(self.max_edits+1))

    def step(self, indices_values, c):
        indices, values = indices_values
        if indices and indices[0] == 0 and values[0] < self.max_edits:
            new_indices = [0]
            new_values = [values[0] + 1]
        else:
            new_indices = []
            new_values = []

        for j,i in enumerate(indices):
            if i == len(self.string): break
            cost = 0 if self.string[i] == c else 1
            val = values[j] + cost
            if new_indices and new_indices[-1] == i:
                val = min(val, new_values[-1] + 1)
            if j+1 < len(indices) and indices[j+1] == i+1:
                val = min(val, values[j+1] + 1)
            if val <= self.max_edits:
                new_indices.append(i+1)
                new_values.append(val)

        return (new_indices, new_values)

    def is_match(self, indices_values):
        indices, values = indices_values
        return bool(indices) and indices[-1] == len(self.string)

    def can_match(self, indices_values):
        indices, values = indices_values
        return bool(indices)

    def transitions(self, indices_values):
        indices, values = indices_values
        return set(self.string[i] for i in indices if i < len(self.string))

    # Calling this method only makes sense if you already checked that the target sentence matches!
    def distance(self, indices_values):
        indices, values = indices_values
        if indices[-1] == len(self.string) and values:
            return values[-1] # the bottom right corner of the sparse matrix
        return False

def distance_within(s1, s2, max_distance):
    "Return the Levenshtein distance between s1 and s2, or False if the distance is more than max_distance."
    aut = LevenshteinAutomaton(s1, max_distance)
    state = aut.start()
    for c in s2:
        state = aut.step(state, c)
        if not aut.can_match(state):
            return False
    if aut.is_match(state):
        return aut.distance(state)
    return False
