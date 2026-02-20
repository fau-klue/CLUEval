# CLUEval

CLUEval is a simple Python module for evaluating text anonymisation using token classification. It provides common metrics such as Precision, Recall and F1-score with the options for a more lenient evaluation.

## Installation
```sh
pip install git+https://github.com/fau-klue/CLUEval
```
### Requirements
- pandas
- numpy
- networkx

## Features

### Join multihead classification 
- Combine spans from multiple classification headers into a single span via an adjacency matrix
- Head labels will be determined by majority voting

### Lenient evaluation
- CLUEval also allows lenient evaluation and accept consider following overlap cases as true positive:
  - Subset: The reference span is contained in candidate.
  - Tiling: The reference span matches multiple adjacent candidates exactly.
  - Overlap: The reference span overlaps several adjacent candidates but should not exceed the length of combined candidate spans.  
- Lenient level:
  - 0: No lenient (Default)
  - 1: Subset
  - 2: Subset + tiling
  - 3: Subset + tiling + overlap

### Metrics
- Precision, Recall and F1
- Span-wise evaluation
  - Compute evaluation metrics without taking the span label into account 
  - Compare spans according to the aforementioned lenient level
- Categorical span-wise evaluation
  - Information category is also considered in the evaluation
  
### Table for error analysis
- CLUEval provides a table for error analysis with colour coded text spans
  - Green (🟩): Tokens occur in both reference and candidate.
  - Red (🟥): Tokens occur in reference but are missing in candidate.
  - Orange (🟧): Tokens appear only in candidate span. 
- Option to input the window size of context information

## Usage
CLUEval expects input data in vertical format (VRT) with BIO tagging scheme.

```python
# fiktives-urteil-p1.bio
# 
# ----------	O	O	O
# AMTSGERICHT	B-anon	B-court-name	B-niedrig
# ERLANGEN	I-anon	I-court-name	I-niedrig
# ----------	O	O	O
```
Further meta information can be included in the VRT file, such as predefined token IDs, document IDs and text domains. 

```python
# reference.bio
# 
# ----------	O	token_0	fictitious_1512	Fictitious_Domain
# AMTSGERICHT	B-niedrig	token_1	fictitious_1512	Fictitious_Domain
# ERLANGEN	I-niedrig	token_2	fictitious_1512	Fictitious_Domain
# ----------	O	token_3	fictitious_1512	Fictitious_Domain
```

### cluevaluate executable script
```
positional arguments:
  reference             Path to reference file.
  candidate             Path to candidate or prediction file.

options:
  -h, --help            show this help message and exit
  -v, --version         output version information and exit
  -a ANNOTATION_LAYER [ANNOTATION_LAYER ...], --annotation_layer ANNOTATION_LAYER [ANNOTATION_LAYER ...]
                        Input names for annotation layers. (default: None)
  -cd DOMAIN_COLUMN, --domain_column DOMAIN_COLUMN
                        Column index of domain information. (default: None)
  -ci DOC_ID_COLUMN, --doc_id_column DOC_ID_COLUMN
                        Column index of document ID (default: None)
  -ct TOKEN_ID_COLUMN, --token_id_column TOKEN_ID_COLUMN
                        Column index of token ids. (default: None)
  -e [{subset,unmatch,overlap} ...], --error_type [{subset,unmatch,overlap} ...]
                        Filter spans by error types and return error tables. 
                        Default setting is 'unmatch' when the option is passed without any addiitonal parameters. 
                        (default: None)
  -l {0,1,2,3}, --lenient_level {0,1,2,3}
                        Level of lenient evaluation. (default: 0)
  -le, --label_evaluation
                        Compute metrics for each category. (default: False)
  -lc LABEL_COLUMN [LABEL_COLUMN ...], --label_column LABEL_COLUMN [LABEL_COLUMN ...]
                        Column name for label values. (default: None)
  -m, --match_tables    Optional argument to print precision and recall matching tables. (default: False)
  -n N_ROW, --n_row N_ROW
                        Number of rows to print. (default: 5)
  -sl SPAN_LABEL_COLUMN, --span_label_column SPAN_LABEL_COLUMN
                        Label column for span-label evaluation. (default: None)
  -slv SPAN_LABEL_VALUE, --span_label_value SPAN_LABEL_VALUE
                        Value to filter label column. (default: None)
  -w, --write_to_file   Argument to write tables to files. (default: False)

```
#### Examples
#### Single layer evaluation
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags
```
#### Multi-layer evaluation
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags pos_tags
```
#### Span evaluation filtered by label column
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags pos_tags -sl ner_tags -slv PERSON
```
#### Include categorical label evaluation
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags pos_tags -le -lc pos_tags
```
#### Include Precision and Recall tables
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags pos_tags -m
```
#### Include error tables
NB: 'unmatch' is the default setting when the `-e` argument is passed without any additional parameters.
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags pos_tags -e unmatch subset
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
# |       2 |     3 | token_1          | token_2        | AMTSGERICHT ERLANGEN               | fictitious_1512 | fictitious_domain | niedrig      | id000001 |
# |       5 |     9 | token_4          | token_8        | Mozartstraße 23 , 91052 Erlangen   | fictitious_1512 | fictitious_domain | hoch         | id000002 |
# |      10 |    11 | token_9          | token_10       | HELGA SCHMIDT                      | fictitious_1512 | fictitious_domain | hoch         | id000003 |
# |      13 |    17 | token_12         | token_16       | Schillerstraße 4 , 91058 Erlangen  | fictitious_1512 | fictitious_domain | hoch         | id000004 |
# |      18 |    21 | token_17         | token_20       | Rechtsanwälte Schneider & Kollegen | fictitious_1512 | fictitious_domain | mittel       | id000005 |
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
# |       2 |     3 |                |              | AMTSGERICHT ERLANGEN             |        |        | anon   | court-name   | niedrig |                    |                  |                                  |          |              |          | unmatch  |      -100 |    -100 |
# |       7 |     9 |                |              | 11 C 122/20                      |        |        | anon   | court-docket | niedrig |                    |                  | 11 C 122/20                      | anon     | court-docket | niedrig  | exact    |         7 |       9 |
# |      10 |    14 |                |              | Mozartstraße 23 , 91052 Erlangen |        |        | anon   | address-name | hoch    |                    |                  | Mozartstraße 23 , 91052 Erlangen | anon     | address-name | hoch     | exact    |        10 |      14 |
# |      17 |    21 |                |              | 09131 / 782 - 01                 |        |        | anon   | code-idx     | niedrig |                    |                  | 09131 / 782 - 01                 | anon     | code-idx     | niedrig  | exact    |        17 |      21 |
# |      24 |    28 |                |              | 09131 / 782 - 105                |        |        | anon   | code-idx     | niedrig |                    |                  | 09131 / 782 - 105                | anon     | code-idx     | niedrig  | exact    |        24 |      28 |
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
##### Columns
`token_id_start` : Predefined token ID for span start token in reference corpus<br>
`token_id_end`: Predefined token ID for span end token in reference corpus<br>
`token_id_start_Y`: Predefined token ID for span start token in candidate corpus<br>
`token_id_end_Y`: Predefined token ID for span end token in candiate corpus<br>
`domain`: Text domain <br>
`reference`: Reference span <br>
`candidate`: Candidate text span <br>
`context`: Highlighted spans + context tokens <br>
`error_type`: Types of matching error <br>
and <br>
input `annotation layers`: Tag columns (here: entity and risk)

##### Examples for possible cases
**Subset**
```
Reference: 21. 05. 2020
Candidate: 21. 05. 2020 und
Error: subset
Context: [...] durch Richterin am Amtsgericht Arnold aufgrund der mündlichen Verhandlungen vom 🟩21. 05. 2020🟩 🟧und🟧 25. 06. 2020 folgendes
```

**Unmatch**

```
Reference: AMTSGERICHT ERLANGEN
Candidate: ---
Error: unmatch
Context: ---------- 🟥AMTSGERICHT ERLANGEN🟥 ----------

Reference: Feldstraße 4 d , 91096 Möhrendorf
Candidate: Feldstraße 4 | , 91096 Möhrendorf
Error: unmatch
Context: LUISE SCHÜTZ , 🟩Feldstraße 4🟩 🟥d🟥 🟩, 91096 Möhrendorf🟩
```