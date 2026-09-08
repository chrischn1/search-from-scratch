"""A search engine written from scratch with numpy only.

Three rankers, all implemented here rather than imported:

  bm25      classic lexical ranking. Matches the words you typed.
  lsa       latent semantic analysis. Truncated SVD of the TF-IDF matrix,
            so a query can match a document that never uses the query word.
  hybrid    normalised sum of the two. Default.

Usage:
    python3 search.py "chinese diaspora"
    python3 search.py --corpus ~/PycharmProjects/knowledge-base/wiki "chrysostom"
    python3 search.py --mode lsa --k 5 "colonial language policy"
"""

import argparse
import math
import pathlib
import re
import sys
from collections import Counter

import numpy as np

# ----------------------------------------------------------------- tokenising

STOP = set("""a an the and or but if while of to in on at by for with from as is
are was were be been being it its this that these those i you he she they we
not no nor so than then there here what which who whom into over under about
""".split())

WORD = re.compile(r"[a-z][a-z']+")
MARKUP = re.compile(r"(```.*?```|`[^`]*`|\[\[|\]\]|[#*_>\-|])", re.S)


def tokenize(text):
    """Markdown in, list of word stems out. The stemmer is deliberately crude."""
    text = MARKUP.sub(" ", text.lower())
    out = []
    for w in WORD.findall(text):
        if w in STOP or len(w) < 3:
            continue
        out.append(stem(w))
    return out


def stem(w):
    """Suffix stripping, enough to fuse plural and participle forms."""
    for suf in ("ational", "iveness", "fulness", "ousness", "ization", "ation",
                "ments", "ness", "ing", "ies", "ment", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            base = w[: -len(suf)]
            if suf == "ies":
                base += "y"
            return base
    return w


# ------------------------------------------------------------------- indexing

class Index:
    def __init__(self, docs, names, dims=64):
        self.names = names
        self.raw = docs
        toks = [tokenize(d) for d in docs]
        self.lengths = np.array([len(t) for t in toks], dtype=float)
        self.avglen = max(self.lengths.mean(), 1.0)

        vocab = {}
        for t in toks:
            for w in t:
                if w not in vocab:
                    vocab[w] = len(vocab)
        self.vocab = vocab

        n_docs, n_terms = len(docs), len(vocab)
        self.tf = np.zeros((n_docs, n_terms), dtype=np.float32)
        for i, t in enumerate(toks):
            for w, c in Counter(t).items():
                self.tf[i, vocab[w]] = c

        df = (self.tf > 0).sum(axis=0)
        # BM25 idf, with the usual +0.5 smoothing so a term in every document
        # scores near zero instead of going negative.
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # TF-IDF with sublinear term frequency, then L2 rows, then SVD.
        tfidf = _sublinear(self.tf) * self.idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        self.tfidf = tfidf / np.maximum(norms, 1e-9)

        dims = int(min(dims, n_docs - 1, n_terms - 1))
        if dims < 2:
            self.U = self.S = self.Vt = None
        else:
            U, S, Vt = np.linalg.svd(self.tfidf, full_matrices=False)
            self.U, self.S, self.Vt = U[:, :dims], S[:dims], Vt[:dims]
            self.doc_vecs = _unit(self.U * self.S)

    # ------------------------------------------------------------- retrieval

    def query_counts(self, q):
        v = np.zeros(len(self.vocab), dtype=np.float32)
        hits = 0
        for w in tokenize(q):
            j = self.vocab.get(w)
            if j is not None:
                v[j] += 1
                hits += 1
        return v, hits

    def bm25(self, q, k1=1.5, b=0.75):
        qv, _ = self.query_counts(q)
        terms = np.nonzero(qv)[0]
        if terms.size == 0:
            return np.zeros(len(self.names))
        f = self.tf[:, terms]
        denom = f + k1 * (1 - b + b * (self.lengths / self.avglen))[:, None]
        return (self.idf[terms] * (f * (k1 + 1)) / np.maximum(denom, 1e-9)).sum(axis=1)

    def lsa(self, q):
        if self.U is None:
            return np.zeros(len(self.names))
        qv, hits = self.query_counts(q)
        if hits == 0:
            return np.zeros(len(self.names))
        qv = _sublinear(qv) * self.idf
        qv = qv / max(np.linalg.norm(qv), 1e-9)
        # Fold the query into the same latent space the documents live in.
        proj = _unit((self.Vt @ qv)[None, :])[0]
        return self.doc_vecs @ proj

    def search(self, q, mode="hybrid", k=5):
        if mode == "bm25":
            score = _norm(self.bm25(q))
        elif mode == "lsa":
            score = _norm(self.lsa(q))
        else:
            score = _norm(self.bm25(q)) + _norm(self.lsa(q))
        order = np.argsort(-score)[:k]
        return [(self.names[i], float(score[i]), snippet(self.raw[i], q))
                for i in order if score[i] > 0]


def _sublinear(counts):
    """1 + log(count), and zero where the count is zero, without log(0)."""
    out = np.zeros_like(counts, dtype=np.float32)
    nz = counts > 0
    out[nz] = 1 + np.log(counts[nz])
    return out


def _unit(m):
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def _norm(v):
    top = v.max()
    return v / top if top > 0 else v


def snippet(text, q, width=180):
    """First line of the document that carries a query stem, else its opening."""
    wanted = set(tokenize(q))
    for line in text.splitlines():
        if len(line.strip()) > 40 and wanted & set(tokenize(line)):
            return line.strip()[:width]
    return " ".join(text.split())[:width]


# ----------------------------------------------------------------------- main

def load(folder):
    docs, names = [], []
    for p in sorted(pathlib.Path(folder).rglob("*.md")):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(t.split()) < 20:
            continue
        docs.append(t)
        names.append(str(p))
    return docs, names


def main():
    ap = argparse.ArgumentParser(description="from-scratch search over markdown")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--corpus", default=str(pathlib.Path(__file__).parent / "corpus"))
    ap.add_argument("--mode", default="hybrid", choices=["bm25", "lsa", "hybrid"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dims", type=int, default=64)
    a = ap.parse_args()

    docs, names = load(a.corpus)
    if not docs:
        sys.exit(f"no markdown files under {a.corpus}")
    idx = Index(docs, names, dims=a.dims)
    q = " ".join(a.query)
    print(f"{len(docs)} documents, {len(idx.vocab)} terms, mode={a.mode}\n")
    hits = idx.search(q, mode=a.mode, k=a.k)
    if not hits:
        print("no match")
    for rank, (name, score, snip) in enumerate(hits, 1):
        print(f"{rank}. {score:.3f}  {pathlib.Path(name).name}")
        print(f"    {snip}\n")


if __name__ == "__main__":
    main()
