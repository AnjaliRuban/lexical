from collections import namedtuple
import json

class Vocab(object):
    PAD = '<pad>'
    SOS = '<s>'
    EOS = '</s>'
    COPY = '<copy>'
    UNK = '<unk>'

    def __init__(self):
        self._contents = {}
        self._unk_contents = {}
        self._rev_contents = {}
        self._unk_rev_contents = {}
        self.add(self.PAD)
        self.add(self.SOS)
        self.add(self.EOS)
        self.add(self.COPY)
        self.add(self.UNK)
        self.train=True

    def add(self, sym):
        if sym not in self._contents:
            i = len(self._contents)
            self._contents[sym] = i
            self._rev_contents[i] = sym
        return self

    def unk_add(self, sym):
        if sym not in self._unk_contents:
            i = len(self._contents) + len(self._unk_contents)
            self._unk_contents[sym] = i
            self._unk_rev_contents[i] = sym
        return self

    def merge(self, add_vocab):
        for sym in add_vocab._contents.keys():
            self.add(sym)
        return self

    def __getitem__(self, sym):
        return self._contents[sym]

    def __contains__(self, sym):
        return sym in self._contents

    def __len__(self):
        return len(self._contents)

    def full_len(self):
        return len(self._contents) + len(self._unk_contents)

    def encode(self, seq, unk=True):
        seq = [s.lower() for s in seq]
        if unk:
            seq = [s if s in self else self.UNK for s in seq]
            e = [self[i] for i in seq]
        else:
            if self.train:
                [self.unk_add(s) for s in seq if not s in self]
            e = [self[i] if i in self else self._unk_contents[i] if i in self._unk_contents else self.unk() for i in seq]
        return e

    def decode(self, seq, unk=True):
        if unk==True:
            return [self._rev_contents[i] for i in seq]
        else:
            return [self._rev_contents[i] if i in self._rev_contents else self._unk_rev_contents[i] for i in seq]

    def get(self, i):
        return self._rev_contents[i]

    def pad(self):
        return self._contents[self.PAD]

    def sos(self):
        return self._contents[self.SOS]

    def eos(self):
        return self._contents[self.EOS]

    def copy(self):
        return self._contents[self.COPY]

    def unk(self):
        return self._contents[self.UNK]

    def __str__(self):
        out = (
            ["Vocab("]
            + ["\t%s:\t%s" % pair for pair in self._contents.items()]
            + [")"]
        )
        return "\n".join(out)

    def dump(self, writer):
        json.dump(self._contents, writer)

    def load(self, reader):
        new_contents = json.load(reader)
        for k, v in new_contents.items():
            if k in self._contents:
                assert self._contents[k] == v
            self._contents[k] = v
            self._rev_contents[v] = k
