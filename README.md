# CLUEval

CLUEval is a lightweight Python module for evaluating text anonymisation using token classification. It provides common metrics such as precision, recall and F1-score with the options for weighting recall more heavily if required.

## Installation
```sh
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
```
#### Example
```sh
cluevaluate <REFERENCE> <PREDICTION>
```
