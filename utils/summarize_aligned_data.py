import sys
from collections import Counter
import pdb
import pickle, json
SPLIT_TOK=" ||| "
def main(data_file, aligner_file):
    data = None
    with open(data_file, "r") as f:
        data = f.read().splitlines()
    print(len(data))
    data = [d for d in data if d.split('|||')[0].strip() != "" and d.split('|||')[1].strip() != ""  and d.split('|||')[0].strip()[:4] != "CLD2" and len(d.split('|||')[0].strip().split(' '))/len(d.split('|||')[1].strip().split(' ')) < 2 and len(d.split('|||')[1].strip().split(' '))/len(d.split('|||')[0].strip().split(' ')) < 2]
    print(len(data))
    word_adjacency = {}
    with open(aligner_file, "r") as f:
        for k, line in enumerate(f):
            input, output = data[k].split(SPLIT_TOK)
            input, output = input.strip().split(" "), output.strip().split(" ")
            alignments = line.strip().split(" ")
            print(alignments)
            print(output)
            for a in alignments:
                if len(a) == 0: continue
                i, j = a.split("-")
                wi = input[int(i)-1]
                wj = output[int(j)-1]
                if wi in word_adjacency:
                    word_adjacency[wi].append(wj)
                else:
                    word_adjacency[wi] = [wj]

    word_alignment = {}
    for (k,v) in word_adjacency.items():
        if len(v) == 0:
            word_alignment[k] = {k:0}
        else:
            word_alignment[k] = dict(Counter(v))

    # with open(aligner_file + '.pickle', 'wb') as handle:
    #     pickle.dump(word_alignment, handle)
    with open(aligner_file + '.json', 'w') as handle:
        json.dump(word_alignment, handle)



if __name__ == "__main__":
    # execute only if run as a script
    assert len(sys.argv) > 2
    main(sys.argv[1], sys.argv[2])
