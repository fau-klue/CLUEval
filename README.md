# CLUEval

CLUEval is a simple Python module for evaluating text anonymisation using token classification. It provides common metrics such as precision, recall and F1-score with the options for weighting recall more heavily if required.

## Installation
```sh
pip install git+https://github.com/fau-klue/CLUEval
```
### Requirements
- pandas
- numpy

## Usage
The input for CLUEval should be formatted according to the BIO tagging scheme.
### Using the clueval executable script
```sh
cluevaluate -h
usage: cluevaluate [-h] reference candidate

positional arguments:
  reference   Path to reference file.
  candidate   Path to candidate or prediction file.

options:
  -h, --help  show this help message and exit
  -v, --version         output version information and exit
  -a ANNOTATION_LAYERS [ANNOTATION_LAYERS ...], --annotation_layers ANNOTATION_LAYERS [ANNOTATION_LAYERS ...]
                        Input names for annotation layers. (default: None)
  -d DOMAIN_COLUMN, --domain_column DOMAIN_COLUMN
                        Column ID for domain information. (default: None)
  -fc FILTER_COLUMN, --filter_column FILTER_COLUMN
                        Column name for filtering. (default: None)
  -fv FILTER_VALUE, --filter_value FILTER_VALUE
                        Filter column by value. (default: None)
  -ce, --categorical_eval
                        Compute metrics for each category. (default: False)
```
### Examples
#### Single layer evaluation
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags
```
#### Multi-layer evaluation
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags ner_tags2
```
#### Conditional evaluation
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags ner_tags2 -fc ner_tags2 -fv PERSON
```
#### Detailed evaluation
```sh
cluevaluate <REFERENCE> <PREDICTION> -a ner_tags ner_tags2 --ce
```