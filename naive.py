import numpy as np
from Bio import SeqIO
import regex
import sys


def compare_all_peptides(p1, p2, k, mismatches=0):
    """Compare all k-mers in sequences :p1 and :p2 with at most :mismatches"""

    if mismatches == 0:
        return sum(p1[i:i + k] in p2 for i in range(len(p1) - k))
    else:
        counter = 0
        for i in range(len(p1) - k):
            pro_pattern = p1[i:i + k]
            counter += len(regex.findall(
                                        "(" + pro_pattern + "){s<=" + str(mismatches) + "}",
                                        p2, overlapped=True))
        return counter


def build_matrix(proteins, k, mismatches=0):
    """Build and return triangle matrix of k-mer overlap for all pairs in :proteins"""

    overlap = np.zeros((len(proteins), len(proteins)))
    for i in range(len(proteins)):
        print("Protein # {} ".format(i + 1), end='', flush=True)
        for j in range(i + 1, len(proteins)):
            print(".", end='', flush=True)
            cross_reactivity = compare_all_peptides(str(proteins[i].seq), str(proteins[j].seq), k,
                                                    mismatches=mismatches)
            overlap[i, j] = cross_reactivity
            overlap[j, i] = cross_reactivity
        print(flush=True)
    return overlap


if __name__ == '__main__':

    fasta = sys.argv[1]
    proteins = [x for x in SeqIO.parse(fasta, "fasta")]
    names = [x.name for x in proteins]

    # k-mer data from length 2 to length 17.
    # MHC class I peptides are usually 8-12 aa long.
    # Class II have a binding core of similar length, but may also have extended ends.

    for k in range(2, 18):

        out_fn = "naive/m{}.csv".format(k)
        x = build_matrix(proteins, k, mismatches=0)
        np.savetxt(out_fn, x, delimiter=',', fmt='%10.1f', comments="", header=",".join(names))
