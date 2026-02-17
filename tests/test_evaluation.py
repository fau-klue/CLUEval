from clueval.evaluation import main 



def test_evaluate(p1, p2):
    main(p1, p2, annotation_layer="confidence")
    main(p1, p2, annotation_layer="confidence")
    main(p1, p2, annotation_layer="confidence")
    main(p1, p2, annotation_layer="confidence", categorical_evaluation=True, categorical_head="confidence")


def test_evaluate_same_file(p1):
    main(p1, p1, annotation_layer="confidence")
    main(p1, p1, annotation_layer="confidence")
    main(p1, p1, annotation_layer="confidence")
    main(p1, p1, annotation_layer="confidence", categorical_evaluation=True, categorical_head="confidence")


def test_evaluation_results(p1, p2):
    pass