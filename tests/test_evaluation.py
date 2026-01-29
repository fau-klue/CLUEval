from clueval.evaluation import main 



def test_evaluate(p1, p2):
    main(p1, p2, annotation_layer="head_0")
    main(p1, p2, annotation_layer="head_1")
    main(p1, p2, annotation_layer=["head_0", "head_1", "head_2"])
    main(p1, p2, annotation_layer=["head_0", "head_1", "head_2"], categorical_evaluation=True, categorical_head=["head_1", "head_2"])


def test_evaluate_same_file(p1):
    main(p1, p1, annotation_layer="head_0")
    main(p1, p1, annotation_layer="head_1")
    main(p1, p1, annotation_layer=["head_0", "head_1", "head_2"])
    main(p1, p1, annotation_layer=["head_0", "head_1", "head_2"], categorical_evaluation=True, categorical_head=["head_1", "head_2"])

