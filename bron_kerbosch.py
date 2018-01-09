from collections import defaultdict
import numpy as np
from copy import deepcopy


def find_cliques(graph):
  """Find and return all cliques within a graph."""

  p = set(graph.keys())
  r = set()
  x = set()
  cliques = []

  for v in degeneracy_ordering(graph):
    neighs = graph[v]

    find_cliques_pivot(graph, r.union([v]), p.intersection(neighs), x.intersection(neighs), cliques)

    p.remove(v)
    x.add(v)

  return sorted(cliques, key=lambda x: -len(x))


def find_cliques_including(item, graph):
  """Find and return all cliques in a graph including item."""

  r = set({item})
  p = set(graph[item])
  x = set()
  cliques = []

  for v in list(p):
    neighs = graph[v]

    find_cliques_pivot(graph, r.union([v]), p.intersection(neighs), x.intersection(neighs), cliques)

    p.remove(v)
    x.add(v)

  return sorted(cliques, key=lambda x: -len(x))


def find_cliques_pivot(graph, r, p, x, cliques):
  if len(p) == 0 and len(x) == 0:
    cliques.append(r)
  else:
    u = next(iter(p.union(x)))
    for v in p.difference(graph[u]):
      neighs = graph[v]
      find_cliques_pivot(graph, r.union([v]), p.intersection(neighs), x.intersection(neighs), cliques)
      p.remove(v)
      x.add(v)


def degeneracy_ordering(graph):
  ordering = []
  ordering_set = set()
  degrees = defaultdict(lambda : 0)
  degen = defaultdict(list)
  max_deg = -1
  for v in graph:
    deg = len(graph[v])
    degen[deg].append(v)
    degrees[v] = deg
    if deg > max_deg:
      max_deg = deg

  while True:
    i = 0
    while i <= max_deg:
      if len(degen[i]) != 0:
        break
      i += 1
    else:
      break
    v = degen[i].pop()
    ordering.append(v)
    ordering_set.add(v)
    for w in graph[v]:
      if w not in ordering_set:
        deg = degrees[w]
        degen[deg].remove(w)
        if deg > 0:
          degrees[w] -= 1
          degen[deg - 1].append(w)

  ordering.reverse()
  return ordering


def remove_tips(tip_list, graph):
  """Prune tips in :tip_list from graph :graph."""
  for tip in tip_list:
    del graph[tip]
    for node in graph:
      if tip in graph[node]:
        graph[node].remove(tip)
  return graph


def csv_2_array(fn, trailing=False):
  """Parse immune overlap matrix csv"""
  with open(fn) as dm:
    names = [x.strip() for x in dm.readline().split(sep=',')]
    if not trailing:
      data = np.asarray([list(map(float, x)) for x in
                        [list(map(str.strip, y)) for y in
                        [z.split(sep=',') for z in dm.readlines()]]])
    else:
      data = np.asarray([list(map(float, x)) for x in
                        [list(map(str.strip, y)) for y in
                        [z.split(sep=',')[:-1] for z in dm.readlines()]]])
  return names, data


def partition(orthogonal_graph, clusters):
    """ Build a set of mutually exclusive cliques in graph :orthogonal_graph.
    Iteratively finds the largest clique and prunes the members from the graph."""

    if len(orthogonal_graph) > 0:
        sorted_cliques = find_cliques(orthogonal_graph)
        max_cliques = [s for s in sorted_cliques if len(s) == len(sorted_cliques[0])]
        for clk in max_cliques:
            c = deepcopy(clusters)
            c.append(clk)
            g = remove_tips([tip for tip in clk], deepcopy(orthogonal_graph))
            partition(g, c)
    else:
        global cluster_sets
        cluster_sets.append(clusters)


if __name__ == "__main__":
    k = 5
    names, data = csv_2_array('naive/m{}.csv'.format(k))
    size = len(names)
    orthogonal_graph = {names[i]:
                            [names[j] for j in range(size) if data[i, j] == 0.0 and i != j]
                        for i in range(size)
                        }

    cluster_sets = list()

    partition(orthogonal_graph, [])
