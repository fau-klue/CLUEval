from clueval.evaluation import evaluate



def test_evaluate(p1, p2):
    evaluate(p1, p2, annotation_layer="layer1")
    evaluate(p1, p2, annotation_layer=["layer1", "layer2"])
    evaluate(p1, p2, annotation_layer=["layer1", "layer2"], categorical_evaluation=True)

def test_evaluate_same(p1):
    evaluate(p1, p1, annotation_layer="layer1")
    evaluate(p1, p1, annotation_layer=["layer1", "layer2"])
    evaluate(p1, p1, annotation_layer=["layer1", "layer2"], categorical_evaluation=True)