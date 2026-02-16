from clueval.evaluation import MetricsForSpansAnonymisation

# CLUEval

CLUEval is a simple Python module for evaluating text anonymisation using token classification. It provides common metrics such as Precision, Recall and F1-score with the options for a more lenient evaluation.

## Installation
```sh
pip install git+https://github.com/fau-klue/CLUEval
```
### Requirements
- pandas
- numpy

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
  - Green: Tokens occur in both reference and candidate.
  - Red: Tokens occur in reference but are missing in candidate.
  - Orange: Tokens appear only in candidate span. 
- Option to input the window size of context information

## Usage
CLUEval expects input data in vertical format (VRT) with BIO tagging scheme.

### cluevaluate executable script
```sh
positional arguments:
  reference             Path to reference file.
  candidate             Path to candidate or prediction file.

options:
  -h, --help            show this help message and exit
  -v, --version         output version information and exit
  -a ANNOTATION_LAYERS [ANNOTATION_LAYERS ...], --annotation_layers ANNOTATION_LAYERS [ANNOTATION_LAYERS ...]
                        Input names for annotation layers. (default: None)
  -t TOKEN_ID_COLUMN, --token_id_column TOKEN_ID_COLUMN
                        Column name for token ids. (default: None)
  -d DOMAIN_COLUMN, --domain_column DOMAIN_COLUMN
                        Column ID for domain information. (default: None)
  -i DOC_ID_COLUMN, --doc_id_column DOC_ID_COLUMN
                        Document ID column (default: None)
  -f FILTER_HEAD, --filter_head FILTER_HEAD
                        Column name for filtering. (default: None)
  -hv HEAD_VALUE, --head_value HEAD_VALUE
                        Filter column by value. (default: None)
  -c, --categorical_eval
                        Compute metrics for each category. (default: False)
  -ch CATEGORICAL_HEAD [CATEGORICAL_HEAD ...], --categorical_head CATEGORICAL_HEAD [CATEGORICAL_HEAD ...]
                        Column name for categorical values. (default: None)
  -l {0,1,2,3}, --lenient_level {0,1,2,3}
                        Level of lenient evaluation. (default: 0)
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
#### Conditional evaluation
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags pos_tags -f ner_tags -hv PERSON
```
#### Include categorical evaluation
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags pos_tags -c -ch pos_tags
```
### Module
Instead of using the provided executable script, you can also embed the CLUEval module into your evaluation script / notebook. 
You will need to import:
- `Convert` and `Match` from `spans_table` 
- `MetricsForSpanAnonymisation` as well as `MetricsForCategoricalSpanAnonymisation` from `evaluation`

#### Create span dataframe for reference and prediction files
```python
from spans_table import Convert, Match

ref_converter = Convert(path_to_file="./tests/data/fiktives-urteil-p1.bio",  annotation_layer=["anon", "entity", "risk"])
cand_converter = Convert(path_to_file="./tests/data/fiktives-urteil-p2.bio", annotation_layer=["anon", "entity", "risk"])

reference = ref_converter()
candidate = cand_converter()

# Prepare precision and Recall tables for later use by matching spans between reference and prediction
recall_matching = Match(reference, candidate)
precision_matching = Match(reference, candidate)
```
#### Output dataframe
```python
# Show first 5 rows from recall_matching dataframe
recall_matching.head()

|   start |   end | token_id_start | token_id_end | text                             | doc_id | domain | anon   | entity       | risk    | token_id_start_Y   | token_id_end_Y   | text_Y                           | doc_id_Y   | domain_Y   | anon_Y   | entity_Y     | risk_Y   | status   |   start_Y |   end_Y |
|--------:|------:|:---------------|:-------------|:---------------------------------|:-------|:-------|:-------|:-------------|:--------|:-------------------|:-----------------|:---------------------------------|:-----------|:-----------|:---------|:-------------|:---------|:---------|----------:|--------:|
|       2 |     3 |                |              | AMTSGERICHT ERLANGEN             |        |        | anon   | court-name   | niedrig |                    |                  | AMTSGERICHT ERLANGEN             |            |            | anon     | court-name   | niedrig  | exact    |         2 |       3 |
|       7 |     9 |                |              | 11 C 122/20                      |        |        | anon   | court-docket | niedrig |                    |                  | 11 C 122/20                      |            |            | anon     | court-docket | niedrig  | exact    |         7 |       9 |
|      10 |    14 |                |              | Mozartstraße 23 , 91052 Erlangen |        |        | anon   | address-name | hoch    |                    |                  | Mozartstraße 23 , 91052 Erlangen |            |            | anon     | address-name | hoch     | exact    |        10 |      14 |
|      17 |    21 |                |              | 09131 / 782 - 01                 |        |        | anon   | code-idx     | niedrig |                    |                  | 09131 / 782 - 01                 |            |            | anon     | code-idx     | niedrig  | exact    |        17 |      21 |
|      24 |    28 |                |              | 09131 / 782 - 105                |        |        | anon   | code-idx     | niedrig |                    |                  | 09131 / 782 - 105                |            |            | anon     | code-idx     | niedrig  | exact    |        24 |      28 |
```

#### Evaluation

```python
# Span-wise evaluation
span_metrics = MetricsForSpansAnonymisation(precision_table=precision_table, recall_table=recall_table)
span_metrics(lenient_level=1, row_name="Span Anonymisation")

#              P         R        F1  TP_Precision  ...  FN  FP  Support  row_name
# Span  91.42857  91.42857  91.42857            64  ...   6   6       70      Span
```
```python
# Categorical span evaluation
categorical_metrics = MetricsForCategoricalSpansAnonymisation(precision_table, recall_table, classification_head="risk")
categorical_metrics(lenient_level=0)

#                 P         R        F1  TP_Precision  TP_Recall  FN  FP  Support
# Hoch     96.96970  96.96970  96.96970            32         32   1   1       33
# Mittel   66.66667  66.66667  66.66667             2          2   1   1        3
# Niedrig  81.81818  81.81818  81.81818            27         27   6   6       33
```

#### Error Analysis
You just need to use `ErrorTable` from `error_analysis` to generate a table for error analysis. Additionally, in order to add context information, you need to use `BIOToSentenceParser` from `spans_table` that maps corpus position to corresponding token.

```python
from error_analysis import ErrorTable
from spans_table import BIOToSentenceParser

reference_sentences = BioToSentenceParser("./tests/data/fiktives-urteil-p1.bio")()
# {'token_ids': [[1, 2, 3, 4], ...], 'sents': [['----------', 'AMTSGERICHT', 'ERLANGEN', '----------'], ...]}

# Example table that contains false negative cases (Ignore cases here: 'exact'
fn_table = recall_table(recall_table["status"].isin(["unmatch"]))
error_analysis = ErrorTable(match_table=fn_table, candidate_table=candidate, token_position_sentence_mapping=reference_sentences)
erroneous_table = error_analysis(headers=["entity", "risk", "risk_Y"], windows=10)
erroneous_table.head(1)
```
| token_id_start   | token_id_end   |   token_id_start_Y |   token_id_end_Y | domain   | entity       | risk    | risk_Y   | reference            | candidate   | context                                                                                                                                | error_type   | comment   |
|:-----------------|:---------------|-------------------:|-----------------:|:---------|:-------------|:--------|:---------|:---------------------|:------------|:---------------------------------------------------------------------------------------------------------------------------------------|:-------------|:----------|
| ---              | ---            |               -100 |             -100 | ---      | court-name   | niedrig | ---      | AMTSGERICHT ERLANGEN | ---         | ---------- 🟥AMTSGERICHT ERLANGEN🟥 ----------                                                                                         | unmatch      |           |
