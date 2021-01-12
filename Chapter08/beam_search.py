from math import log

import matplotlib.pyplot as plt
import numpy as np
from numpy import array


def beam_search_decoder(data, k):
    """
    Beam search implemeted
    """
    path = [[list(), 1.0]]
    for row in range(len(data)):
        selected_path = []
        index_score = []
        for i in path:
            sequence, score = i
            for j in range(len(data[row])):
                index_score.append([sequence + [j], score * -log(data[row][j])])
            selected_path.append(index_score)
        sorted_index_score = sorted(index_score, key=lambda tup: tup[1])
        path = sorted_index_score[:k]
    return path


# define a sequence of 10 words over a vocab of 5 words
data = np.random.rand(10, 10)
data = array(data)
# decode sequence
result = beam_search_decoder(data, 3)
# print result
for seq in result:
    print(seq)
plt.xticks(range(5))
plt.yticks(range(10))
plt.imshow(data)
plt.show()

# [[9, 2, 9, 3, 1, 0, 1, 8, 9, 6], 4.1305149239915766e-13]
# [[9, 2, 9, 3, 1, 0, 1, 8, 9, 9], 5.016578875199497e-13]
# [[9, 7, 9, 3, 1, 0, 1, 8, 9, 6], 5.907125818068977e-13]
