# CLUEval

CLUEval is a Python module and command line interface for span-based evaluation of sequence labelling. It expects two sequences of non-overlapping spans, denoted as **R** = reference and **C** = candidate spans. These spans can for instance be gold spans and predictions of an automatic model, or different predictions of automatic models or different labelling of human annotators. CLUEval provides common metrics such as precision, recall, and F1-score, with the possibility of choosing between strict evaluation and different **levels of leniency**.

## Installation
```sh
pip install git+https://github.com/fau-klue/CLUEval
```

### Dependencies
- pandas
- numpy
- networkx

## Input format
CLUEval expects two files with input data in verticalised text format (VRT), where each token is on a separate line and annotated with BIO tags. It assumes that there are at least two columns, the first being the token and the second one the annotation, such as

|           |             |
|-----------|-------------|
| Stephanie | B-sensitive |
| works     | O           |
| at        | O           |
| city      | B-sensitive |
| court     | I-sensitive |
| .         | O           |


It can also contain several different annotation layers, such as

|           |             |              |        |
|-----------|-------------|--------------|--------|
| Stephanie | B-sensitive | B-nat-name   | B-high |
| works     | O           | O            | O      |
| at        | O           | O            | O      |
| city      | B-sensitive | B-court-name | B-low  |
| court     | I-sensitive | I-court-name | I-low  |
| .         | O           | O            | O      |

Further information, such as token IDs, document IDs, or text domains, can be included as token-level annotation following annotation layers:

|           |             |              |        |        |      |       |
|-----------|-------------|--------------|--------|--------|------|-------|
| Stephanie | B-sensitive | B-nat-name   | B-high | token0 | doc0 | legal |
| works     | O           | O            | O      | token1 | doc0 | legal |
| at        | O           | O            | O      | token2 | doc0 | legal |
| city      | B-sensitive | B-court-name | B-low  | token3 | doc0 | legal |
| court     | I-sensitive | I-court-name | I-low  | token4 | doc0 | legal |
| .         | O           | O            | O      | token5 | doc0 | legal |


## Features

### Multilayer span

- Convert tokens to span: 
  - Extract tagged tokens for each annotation layer (Ignore O-tag).
  - Combine consecutive tokens into a whole span.
  - Extract span label by removing B- and I-tags.
- Unify spans across annotation layers:
  - Identify spans that overlap across layers.
  - Merge these spans into a unified span via adjacency matrix.
- Span label:
  - Collect predicted label for each combined span.
  - Assign label according to majority vote.

### Metrics

- Calculation of precision, recall, and F1 (harmonic mean between precision and recall)
- Labelled vs. unlabelled evaluation

### Lenient evaluation

- When calculating **recall**, reference spans are classified in true positives (TP) and false negatives (FN).
- CLUEval allows lenient evaluation, which considers more spans than just exact matches as correct.
- Consider the following reference span (R) vs. different kinds of candidate spans (C):

```
R      |==========|

C      |----------|        0. exact
    |-------------|        1. contained
     
       |---||-----|        2. tiled
     |--------||----|      3. covered
     
    (all other cases)      4. unmatched
```

- We distinguish four types of reference-candidate alignments, defined from the perspective of the reference span, that may be counted as TP:
  - exact match: The reference span and the candidate span are identical.
  - contained match: The reference span is fully contained in a larger candidate span.
  - tiled match: The reference span matches multiple adjacent candidate spans exactly.
  - covered match: The reference span overlaps with several adjacent candidate spans but does not exceed the length of the combined candidate spans.
- The level of leniency determines which kinds of matches are classified as TP:
  - 0: strict evaluation, i.e. only exact matches are classified as TP
  - 1: incl. contained matches
  - 2: incl. contained + tiled matches
  - 3: incl. contained + tiled + covered matches (**default**)
- All other kinds of (partially matched or unmatched) reference spans are counted as false negative (FN).
- The calculation of **precision** is accomplished by calculating recall for reference spans with regard to candidate annotations.

### Precision and recall tables
- Prepare table:
  - Match spans in reference and candidate tables via their **corpus positions** (start and end offsets).
  - Identify all **exact** matches with inner join and keep them in a separate table.
  - For remaining spans, check for lenient matches and assign span status accordingly (see **Lenient evaluation**).
  - Concat all spans into a unified dataframe. 
- **Precision** table: 
  - Match candidate to reference spans.
  - Show how many candidate spans were correct with respect to reference.
- **Recall** table: 
  - Match reference to candidate.
  - Show how many annotated spans are correctly identified.
### Span evaluation
- Evaluate how accurately the model can identify the spans without considering the labels.
- Compare the start and end positions of reference and candidate spans with respect to the lenient level.

### Span evaluation filtered by label column
- Filter spans:
  - Select a label column from reference to filter spans.
  - Specify the column value to retain relevant spans in precision and recall tables for evaluation. 
- Compute span-wise metrics for filtered spans **without comparing the label values in R and C**.

### Labelled evaluation

- Calculate span-wise evaluation metrics with label-matching for each label column separately.
- Only spans that **satisfy the lenient level and match in labels** are considered as true positive (TP).
- All other cases will be counted as false negative (FN), or false positive (FP), respectively.

### Error analysis tables
- CLUEval provides a table for error analysis with colour coded text spans
  - Green (🟩): Tokens occur in both reference and candidate.
  - Red (🟥): Tokens occur in reference but are missing in candidate.
  - Orange (🟧): Tokens appear only in candidate span. 
- Option to input the window size of context information

## Usage

### `cluevaluate` executable script

```
positional arguments:
  reference             Path to reference file.
  candidate             Path to candidate or prediction file.

options:
  -h, --help            show this help message and exit
  -v, --version         output version information and exit
  -l {0,1,2,3}, --lenient {0,1,2,3}
                        Level of leniency. (default: 3)
  -a ANNOTATION_LAYER [ANNOTATION_LAYER ...], --annotation_layer ANNOTATION_LAYER [ANNOTATION_LAYER ...]
                        Names of annotation layers. (default: ['span'])
  -lc LABEL_COLUMN [LABEL_COLUMN ...], --label_column LABEL_COLUMN [LABEL_COLUMN ...]
                        Column name for labelled evaluation. (default: None)
  -sl SPAN_LABEL_COLUMN, --span_label_column SPAN_LABEL_COLUMN
                        Label column for span-label evaluation. (default: None)
  -slv SPAN_LABEL_VALUE, --span_label_value SPAN_LABEL_VALUE
                        Value to filter label column. (default: None)
  -cd DOMAIN_COLUMN, --domain_column DOMAIN_COLUMN
                        Column index of domain information. (default: None)
  -ci DOC_ID_COLUMN, --doc_id_column DOC_ID_COLUMN
                        Column index of document ID (default: None)
  -ct TOKEN_ID_COLUMN, --token_id_column TOKEN_ID_COLUMN
                        Column index of token ids. (default: None)
  -e [{contained,tiled,covered,unmatched} ...], --error_type [{contained,tiled,covered,unmatched} ...]
                        Filter spans by error types and return error tables. Default setting is 'unmatched' when the option is passed without any addiitonal
                        parameters. (default: None)
  -m, --match_tables    Optional argument to print precision and recall matching tables. (default: False)
  -n N_ROW, --n_row N_ROW
                        Number of rows to print. (default: 5)
  -w, --write_to_file   Write tables to files? (Path will be determined automatically) (default: False)
```

#### Examples with fictitious verdict

- Basic span-wise evaluation
```sh
cluevaluate ./tests/data/fiktives-urteil-p1.bio ./tests/data/fiktives-urteil-p2.bio -l 0
```
- Multilayer evaluation
```sh
cluevaluate ./tests/data/fiktives-urteil-p1.bio ./tests/data/fiktives-urteil-p2.bio -a span entity
```
- Labelled evaluation
```sh
cluevaluate ./tests/data/fiktives-urteil-p1.bio ./tests/data/fiktives-urteil-p2.bio -a span entity risk -lc entity risk
```
- Span evaluation filtered by label column
```sh
cluevaluate ./tests/data/fiktives-urteil-p1.bio ./tests/data/fiktives-urteil-p2.bio -a span entity -sl entity -slv nat-name
```
- Include Precision and Recall tables
```sh
cluevaluate ./tests/data/fiktives-urteil-p1.bio ./tests/data/fiktives-urteil-p2.bio -a ner_tags pos_tags -m
```
- Include error tables; NB: 'unmatched' is the default setting when the `-e` argument is passed without any additional parameters.
```sh
cluevaluate ./tests/data/fiktives-urteil-p1.bio ./tests/data/fiktives-urteil-p2.bio -a ner_tags pos_tags -e unmatched contained
```

### Module
Instead of using the provided executable script, you can also embed the CLUEval module into your evaluation script / notebook. 
You will need to import:
- `Convert` and `Match` from `spans_table` 
- `MetricsForSpanAnonymisation` as well as `MetricsForCategoricalSpanAnonymisation` from `evaluation`

#### Convert and match reference and prediction files to span dataframe
```python
from clueval.spans_table import Convert, Match

ref_converter = Convert(path_to_file="./tests/data/fiktives-urteil-p1.bio",  annotation_layer=["anon", "entity", "risk"])
cand_converter = Convert(path_to_file="./tests/data/fiktives-urteil-p2.bio", annotation_layer=["anon", "entity", "risk"])

reference = ref_converter()
candidate = cand_converter()

# Prepare precision and Recall tables for later use by matching spans between reference and prediction
recall_matching = Match(reference, candidate)
precision_matching = Match(reference, candidate)
```

#### Further span meta information
```python
from clueval.spans_table import Convert

ref_converter = Convert(path_to_file="./tests/data/reference.bio", 
                        annotation_layer=["confidence"],
                        token_id_column=2,
                        doc_id_column=3,
                        domain_column=4)
reference = ref_converter()
reference.head()

# |   start |   end | token_id_start   | token_id_end   | text                               | doc_id          | domain            | confidence   | id       |
# |--------:|------:|:-----------------|:---------------|:-----------------------------------|:----------------|:------------------|:-------------|:---------|
# |       1 |     2 | token_1          | token_2        | AMTSGERICHT ERLANGEN               | fictitious_1512 | fictitious_domain | niedrig      | id000001 |
# |       4 |     8 | token_4          | token_8        | Mozartstraße 23 , 91052 Erlangen   | fictitious_1512 | fictitious_domain | hoch         | id000002 |
# |       9 |    10 | token_9          | token_10       | HELGA SCHMIDT                      | fictitious_1512 | fictitious_domain | hoch         | id000003 |
# |      12 |    16 | token_12         | token_16       | Schillerstraße 4 , 91058 Erlangen  | fictitious_1512 | fictitious_domain | hoch         | id000004 |
# |      17 |    20 | token_17         | token_20       | Rechtsanwälte Schneider & Kollegen | fictitious_1512 | fictitious_domain | mittel       | id000005 |
```
##### Columns in Convert dataframe
`start`: Span start position in corpus <br>
`end`: Span end position in corpus <br>
`token_id_start` : Predefined token ID for span start token in corpus<br>
`token_id_end`: Predefined token ID for span end token in corpus<br>
`text`: Extracted text span <br>
`doc_id`: Document Id<br>
`domain`: Text domain <br>
`id`: Span id <br>
input `annotation layers`: Tag columns (here: confidence)

#### Match dataframe
```python
# Show first 5 rows from recall_matching dataframe
recall_matching.head()

# |   start |   end | token_id_start | token_id_end | text                             | doc_id | domain | anon   | entity       | risk    | token_id_start_Y   | token_id_end_Y   | text_Y                           |  anon_Y  | entity_Y     | risk_Y   | status   |   start_Y |   end_Y |
# |--------:|------:|:---------------|:-------------|:---------------------------------|:-------|:-------|:-------|:-------------|:--------|:-------------------|:-----------------|:---------------------------------|:---------|:-------------|:---------|:---------|:----------|:--------|
# |       1 |     2 |                |              | AMTSGERICHT ERLANGEN             |        |        | anon   | court-name   | niedrig |                    |                  |                                  |          |              |          | unmatch  |      -100 |    -100 |
# |       6 |     8 |                |              | 11 C 122/20                      |        |        | anon   | court-docket | niedrig |                    |                  | 11 C 122/20                      | anon     | court-docket | niedrig  | exact    |         7 |       9 |
# |       9 |    13 |                |              | Mozartstraße 23 , 91052 Erlangen |        |        | anon   | address-name | hoch    |                    |                  | Mozartstraße 23 , 91052 Erlangen | anon     | address-name | hoch     | exact    |        10 |      14 |
# |      16 |    20 |                |              | 09131 / 782 - 01                 |        |        | anon   | code-idx     | niedrig |                    |                  | 09131 / 782 - 01                 | anon     | code-idx     | niedrig  | exact    |        17 |      21 |
# |      23 |    27 |                |              | 09131 / 782 - 105                |        |        | anon   | code-idx     | niedrig |                    |                  | 09131 / 782 - 105                | anon     | code-idx     | niedrig  | exact    |        24 |      28 |
```
##### Columns in Match dataframe
`start`: Span start position in reference corpus <br>
`end`: Span end position in reference corpus <br>
`token_id_start` : Predefined token ID for span start token in reference corpus<br>
`token_id_end`: Predefined token ID for span end token in reference corpus<br>
`text`: Extracted text span <br>
`doc_id`: Document Id<br>
`domain`: Text domain <br>
`start_Y`:Span start position in candidate corpus <br>
`end_Y`: Span end position in candidate corpus <br>
`token_id_start_Y`: Predefined token ID for span start token in candidate corpus<br>
`token_id_end_Y`: Predefined token ID for span end token in candiate corpus<br>
`text_Y`: Candidate text span<br>
`status`: Matching types between reference and candidate <br>
and <br>
input `annotation layers`: Tag columns (here: anon, entity, risk)


#### Evaluation

```python
from clueval.evaluation import MetricsForSpansAnonymisation
# Span-wise evaluation
span_metrics = MetricsForSpansAnonymisation(precision_table=precision_table, recall_table=recall_table)
span_metrics(lenient_level=1, row_name="Span Anonymisation")

#              P         R        F1  TP_Precision  ...  FN  FP  Support  row_name
# Span  85.71429  85.71429  85.71429            60  ...  10  10       70      Span
```
```python
from clueval.evaluation import MetricsForCategoricalSpansAnonymisation
# Categorical span evaluation
categorical_metrics = MetricsForCategoricalSpansAnonymisation(precision_table, recall_table, classification_head="risk")
categorical_metrics(lenient_level=0)

#                 P         R        F1  TP_Precision  TP_Recall  FN  FP  Support
# Hoch     78.37838  87.87879  82.85714            29         29   4   8       33
# Mittel   66.66667  66.66667  66.66667             2          2   1   1        3
# Niedrig  92.85714  78.78788  85.24590            26         26   7   2       33
```

#### Error Analysis
You just need to use `ErrorTable` from `error_analysis` to generate a table for error assessment. Additionally, in order to retrieve context information, you need to use `BIOToSentenceParser` from `spans_table` that maps corpus position to corresponding token.

```python
from clueval.error_analysis import ErrorTable
from clueval.spans_table import BIOToSentenceParser

reference_sentences = BioToSentenceParser("./tests/data/fiktives-urteil-p1.bio")()
# {'token_ids': [[1, 2, 3, 4], ...], 'sents': [['----------', 'AMTSGERICHT', 'ERLANGEN', '----------'], ...]}

# Example table that contains false negative cases (Ignore cases here: 'exact'
fn_table = recall_table(recall_table["status"].isin(["unmatch"]))
error_analysis = ErrorTable(match_table=fn_table, candidate_table=candidate, token_position_sentence_mapping=reference_sentences)
erroneous_table = error_analysis(headers=["entity", "risk", "risk_Y"], windows=10)
erroneous_table.head(1)

# | token_id_start   | token_id_end   |   token_id_start_Y |   token_id_end_Y | domain   | entity       | risk    | risk_Y   | reference            | candidate   | context                                                                                                                                | error_type   | comment   |
# |:-----------------|:---------------|-------------------:|-----------------:|:---------|:-------------|:--------|:---------|:---------------------|:------------|:---------------------------------------------------------------------------------------------------------------------------------------|:-------------|:----------|
# | ---              | ---            |               -100 |             -100 | ---      | court-name   | niedrig | ---      | AMTSGERICHT ERLANGEN | ---         | ---------- 🟥AMTSGERICHT ERLANGEN🟥 ----------                                                                                         | unmatch      |           |

```
##### Columns in ErrorTable dataframe
- `token_id_start` : Predefined token ID for span start token in reference corpus
- `token_id_end`: Predefined token ID for span end token in reference corpus
- `token_id_start_Y`: Predefined token ID for span start token in candidate corpus
- `token_id_end_Y`: Predefined token ID for span end token in candiate corpus
- `domain`: Text domain
- `reference`: Reference span
- `candidate`: Candidate text span
- `context`: Highlighted spans + context tokens
- `error_type`: Types of matching error
- `annotation layers`: Tag columns (here: entity and risk)

##### Examples for possible cases
**contained**
```
Reference: 21. 05. 2020
Candidate: 21. 05. 2020 und
Error: contained
Context: [...] durch Richterin am Amtsgericht Arnold aufgrund der mündlichen Verhandlungen vom 🟩21. 05. 2020🟩 🟧und🟧 25. 06. 2020 folgendes
```

**unmatched**

```
Reference: AMTSGERICHT ERLANGEN
Candidate: ---
Error: unmatched
Context: ---------- 🟥AMTSGERICHT ERLANGEN🟥 ----------

Reference: Feldstraße 4 d , 91096 Möhrendorf
Candidate: Feldstraße 4 | , 91096 Möhrendorf
Error: unmatch
Context: LUISE SCHÜTZ , 🟩Feldstraße 4🟩 🟥d🟥 🟩, 91096 Möhrendorf🟩
```
