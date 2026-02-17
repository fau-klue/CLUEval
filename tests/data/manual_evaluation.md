reference:
start
2
5
10
13
18
23
29
35
46
54
75

end
3
9
11
17
21
28
33
39
51
56
79

n spans in reference:
- 11

n spans in candidate:
- 22

recall:
    - exact: TP - 3 FN - 8 -> 0.273
    - lenient 1: TP - 5 FN - 6 -> 0.454
    - lenient 2: TP - 6 FN - 5   -> 0.545
    - lenient 3: TP - 6 FN - 5 -> 0.545

  precision
    - exact: TP - 3 FP - 19 -> 0.1364
    - lenient 1: TP - 18 FP 4 -> 0.8181
    - lenient 2: TP - 18 FN - 4 -> 0.8181
    - lenient 3: TP - 18 FN - 4 -> 0.8181