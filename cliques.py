import numpy as np
from bron_kerbosch import *


with open('full_matrix.csv') as dm:
	names = [x.strip() for x in dm.readline().split(sep=',')]
	data = np.asarray([list(map(float, x)) for x in
						[list(map(str.strip, y)) for y in
						[z.split(sep=',') for z in dm.readlines()]]])

size = len(names)

orthogonal_graph = {names[i]: [names[j] for j in range(size) if data[i, j] == 0.0 and i != j] for i in range(size)}

sorted_cliques = find_cliques_including("AAK33936.1", orthogonal_graph)

print("Found {} cliques of length {}".format(len(sorted_cliques), len(sorted_cliques[0])))
print(sorted_cliques[0])

