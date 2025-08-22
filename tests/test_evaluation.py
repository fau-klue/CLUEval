from clueval.evaluation import evaluate


def test_evaluate(p1, p2):

    evaluate(p1, p2)


def test_evaluate_same(p1):

    evaluate(p1, p1)
