# The CLUEval algorithm

## Evaluation

### Data & assumptions

- input data are two sequences of **non-overlapping spans**, usually **G** = gold spans and **P** = predicted spans; but they could also be predictions of two different models or spans from two human annotators
- algorithms assume that spans within each sequence aren't nested and do not overlap!
- spans are represented by start/end **corpus positions** (i.e. consecutive numbers assigned to tokens) and may be annotated  with one or more attributes (e.g. _information category_ and _risk level_)
- the canonical input format are vertical text files (`.vrt`) with BIO tagging as token-level annotation (and possibly other annotation layers); users should be able to specify which columns contain BIO tags; XML tags provide additional metadata e.g. text ID, legal domain, sentence ID, …; token-level annotation should include unique token IDs for consistency checks and for safe back-references from spans into the original corpus
- import function automatically generates one sequence of spans for each BIO column; multiple sequences can be merged into a single sequence of spans with multiple attributes (see below); corpus positions are automatically determined by counting tokens in the input file (and skipping XML tags)
- optionally, data can also be provided as a table of pre-computed spans, but matching `.vrt` files are still needed to provide sentence contexts for error analysis; this approach is dispreferred because of the risk that corpus positions in the spans table might be inconsistent

### Recall

- the CLUeval algorithm computes recall and precision for spans in an asymmetric way (which makes sense because span detection is not a classification task and their is no meaningful notion of true negatives to complete the contigency table)
- we define **recall** as the proportion of gold spans detected by the model (see below for the precise definition of “detected”, which can be understood in more or less lenient ways); it is thus based on a classification of the spans in **G** into true positives (TP) and false negatives (FN)
- conversely, we define **precision** as the proportion of predicted spans that are supported by the gold standard; it is thus based on a classification of the spans in **P** into true positives (TP) and false positives (FP); note that the support of recall values is thus usually different from the support of precision values, and the number of TPs might also differ
- in fact, we will use the same classification algorithm for recall and precision; it is just applied in different directions
- to determine recall, we classify each **G** span according to how it overlaps with spans from **P**, as illustrated by the table below

```
G      |==========|

P      |----------|        1. exact match
    |-------------|        2. superset
     
       |---||-----|        3. tiling
     |--------||----|      4. overlap
     
    (all other cases)      5. FN
```

- it is obvious that both 1. and 2. should count as TPs when computing recall (because all sensitive information in the span is masked); cases 3. and 4. are less obvious, but can be justified as TPs when computing unlabelled recall, i.e. if the goal is to determine whether the model is capable of masking all sensitive information; for labelled recall, it might be better to treat them as FNs as the labels of the component spans in **P** might be different and thus cannot be matched uniquely with **G**
- if we only accept 1. as TPs, we should obtain the same recall and precision values as `seqeval`; the evaluation function should offer users a choice to compute unlabelled or labelled recall and how lenient the evaluation should be (levels 1--4); for lenient evaluation of labelled recall, heuristics will be used to determine the labels of joined spans
- sketch of the **classification algorithm** `match.spans(G, P)`, which operates on Pandas data frames
  - find all exact span matches, assign to class 1., and remove them temporarily from both **G** and **P** (because they cannot possibly be part of any overlap matches) ➞ remaining spans **G'** and **P'**
  - then iterate over all **G'** spans (s0, e0):
    - find a possible superset span (s, e), i.e. which satisfies `s <= s0 & e >= e0` by a single Pandas query on the **P'** data frame; this can return at most one match; if found, assign to class 2. and continue, but do *not* remove (s, e) from **P'**
    - otherwise find all overlapping spans (s, e) in **P'** with the query `~(e < s0 | e0 < s)` (i.e. neither P span is completely before G span, nor G span completely before P span), and make sure they're sorted by corpus position
    - check if the overlapping spans are immediately adjacent; if not, assign to class 5. and continue
    - join the overlapping spans into a single span (s', e'), and determine a combined label for the new span (based on total number of tokens for each label that overlap with (s0, e0))
    - if `s' == s0 && e' == e0` assign to class 3. and continue
    - if `s' <= s0 && e' >= e0` assign to class 4. and continue
    - in all other cases, assign to class 5., i.e. we classify them as FN even if there is partial overlap
  - recombine the classification of **G'** with the exact matches, making sure that all spans occur in correct order; it is sufficient to return just the series of classes (which can then be added as a new column to **G**)

### Precision

- precision is based on a classification of all **P** spans, which are considered TPs if they match a **G** span exactly or are contained in one, and additionally if they are completely covered by **G** spans for unlabelled precision (because none of the tokens detected by the model are incorrect)
- we use exactly the same classification algorithm as for recall, just in the opposite direction: `match.spans(P, G)`
- in other words, precision of **P** is the same as the recall of **G** on **P**

### Error analysis

- for manual validation of remaining errors and qualitative analysis, we need tables of all FNs and FPs which show the problematic spans in full sentence context, with all relevant **G** and **P** spans marked
- in contrast to earlier evaluations, the error table and context display cannot be based on `match.spans()` because partial overlaps also need to be included in the display (which are all simply classified as FN for the evaluation)
- we thus need a separate function that takes either a list of FNs or a list of FPs and generates the corresponding entries for the error table by looking up relevant information directly in **G** and **P** (and using the underlying `.vrt` tables to render sentence contexts)
- again, we can use the same function for both cases: `error_table(FN_spans, G, P)` for the FNs, and `error_table(FP_spans, P, G)` for the FPs
- for each problematic span in **G**, the error table contains a class (tiling, overlap, partial, FN), the surface string of the span, its label(s), its start and end cpos and token ID; as well as the combined surface string of all overlapping spans in **P**, their combined labels, and the minimum start and maximum end cpos across all overlapping spans, with corresponding token IDs --- and, of course, vice versa for FPs
- this is obtained by iterating over the list of FNs (or FPs) and finding all overlapping spans in **P** (or **G**) with the same Pandas query as above (`~(e < s0 | e0 < s)`), sorting them by cpos
- in addition, the problematic span is shown in its full sentence context, highlighting both the **G** span and all overlapping **P** spans in a suitable way; this can be done most nicely with different colours in a HTML rendering (also including tooltips with labels for the spans); for manual annotation in a spreadsheet editor, it can be simulated with colorful emoji
- note the asymmetry in the context display: for a FN, we only show a single **G** span together with all overlapping **P** spans; for a FP, we show a single **P** span with all overlapping **G** spans; everything else would make lookup and display too complex

## Combined model

- if there are multiple BIO columns in the input file (e.g. because spans are annotated with multiple attributes), it would be desirable to merge them into a single sequence of spans labelled with multiple attributes; this often improves recall in cases where a span has been missed by one of the BIO taggers, but found by the other ones
- because it is theoretically possible for such spans to overlap in complex ways, merging spans is far from trivial; our previous implementations have used complicated and somewhat unsystematic heuristics, typically starting from one of the  columns and only extending/adding spans in certain cases
- we should instead have a consistent, well-defined and symmetric algorithm for matching spans
- the core idea is to find complete sets of overlapping spans across all BIO layers and merge them into a single span; with heuristics for choosing majority labels in case multiple spans from the same layer are involved
- in mathematical terms, we take the union of all spans from all BIO layers as our base set and divide it into **overlap components**
- this is achieved easily via a graph algorithm:
  - the union of all spans form the nodes of the graph
  - we draw an edge between any two overlapping spans (including exact matches); for this purpose, we have to check all possible pairs of spans for overlap to obtain the full adjacency matrix
  - we can then use a standard algorithm (from `networkx`) to divide the graph into **connected components**, which are in fact overlap components because of how the adjacency matrix was computed
- for better efficiency, this step can be carried out separately for each document (as the adjacency matrix has quadratic complexity)
- each overlap components is then merged into a single span that contains all spans in the component (i.e. min over start positions and max over end positions); majority labels for the merged span are determined by counting how many tokens of the span are labelled accordingly
- for the (hopefully) common case where all BIO layers contain exactly the same span (with their different labels), the algorithm does the right thing automatically, so no special cases are needed



