## Check span start and end positions
#### reference
###### start
2 5 10 13 18 23 29 35 46 54 75

###### end
3 9 11 17 21 28 33 39 51 56 79

## Evaluation
### Span 
n spans in reference:
- 11

n spans in candidate:
- 22

#### Recall
- exact: TP - 3 FN - 8 -> 0.273
- lenient 1: TP - 5 FN - 6 -> 0.454
- lenient 2: TP - 6 FN - 5   -> 0.545
- lenient 3: TP - 6 FN - 5 -> 0.545

#### Precision
- exact: TP - 3 FP - 19 -> 0.1364
- lenient 1: TP - 18 FP 4 -> 0.8181
- lenient 2: TP - 18 FN - 4 -> 0.8181
- lenient 3: TP - 18 FN - 4 -> 0.8181
    
### Categorical span
- categories: Hoch, Mittel, Niedrig
##### Number of span
- Hoch: 
  - ref: 7
  - cand: 12
- Mittel: 
  - ref: 2
  - cand: 6
- Niedrig: 
  - ref: 2
  - cand: 4

##### Recall
- Hoch
  - exact: TP - 1 FN - 6
  - lenient 1: TP - 2 FN - 5
  - lenient 2: TP - 3 FN - 4
  - lenient 3:  TP - 3 FN - 4

- Mittel
  - exact: TP - 1 FN - 1
  - lenient 1: TP - 1 FN - 1
  - lenient 2: TP - 1 FN - 1
  - lenient 3:  TP - 1 FN - 1

- Niedrig
  - exact: TP - 1 FN - 1
  - lenient 1: TP - 2 FN - 0
  - lenient 2: TP - 2 FN - 0
  - lenient 3:  TP - 2 FN - 

#### Precision
- Hoch
  - exact: TP - 1 FN - 11
  - lenient 1: TP - 10 FN 2
  - lenient 2: TP - 10 FN 2
  - lenient 3: TP -10 FN 2

- Mittel
  - exact: TP - 1 FN - 5
  - lenient 1: TP - 6 FN - 0
  - lenient 2: TP - 6 FN - 0
  - lenient 3:  TP - 6 FN - 0

- Niedrig
  - exact: TP - 1 FN - 3
  - lenient 1: TP - 2 FN - 2
  - lenient 2: TP - 2 FN - 2
  - lenient 3:  TP - 2 FN - 2
