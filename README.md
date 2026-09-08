# search-from-scratch

A document search engine written in one file with numpy as the only dependency.
No search library, no embedding API, no vector database. The ranking functions,
the tokeniser, the inverted term matrix and the dimensionality reduction are all
implemented here so that every scoring decision is visible and can be changed.

A better local search box than the shitty one built in, it even helps when you ponder 
what do I do when the document I want never uses the word I typed. 
A lexical ranker scores a document
by the query words it contains, so a page about canon formation is invisible to
the query "reading list chosen by teachers". A latent semantic ranker scores it
by position in a reduced concept space built from the whole corpus, so that page
comes back first. Running both and adding the results keeps the precision of the
lexical match and adds the recall of the semantic one.

## Run it

```bash
python3 search.py "how do speakers persuade an audience"
python3 search.py --mode lsa --k 3 "reading list chosen by teachers"
python3 search.py --corpus ~/notes --mode bm25 "citation"
```

`--mode` selects `bm25`, `lsa` or `hybrid`, and `hybrid` is the default.
`--corpus` points at any folder of Markdown files, searched recursively.
`--k` sets how many results to print and `--dims` sets the size of the concept
space.

## What is inside

The tokeniser lowercases the text, strips Markdown, drops a stop word list and
runs a crude suffix stripper so that "translating", "translation" and
"translations" collapse to one term.

BM25 is the lexical ranker. It weights a term by inverse document frequency,
saturates repeated occurrences through the k1 parameter so that a word used
thirty times does not score thirty times a word used once, and normalises by
document length through the b parameter so that long documents do not win by
size alone.

Latent semantic analysis is the second ranker. The corpus becomes a TF-IDF
matrix with sublinear term frequency and L2 normalised rows, and a singular value
decomposition truncated to the first k singular values gives every document a
short dense vector. The query is folded into the same space through the term
matrix and ranked by cosine similarity. Terms that co-occur across the corpus end
up close together, which is why a query can match a document that shares no words
with it.

The hybrid mode divides each score vector by its own maximum before adding, so
that the two rankers, which are on unrelated scales, contribute comparably.

## Scale

The whole index is a dense matrix, which is the honest limit of this design. On
461 Markdown files and 14,211 terms it indexes and answers in about one second on
a laptop. A corpus of a hundred thousand documents needs a sparse matrix and a
randomised SVD, and at that point the interesting part of the exercise is over.

## Corpus

`corpus/` holds eight short notes written for this repository so that the tool
runs with no setup. Point `--corpus` at your own folder for real use.
