from collections import Counter

def majority_vote(labels: list[str]):
    """
    Determine the most common label from list of labels
    :param labels: List of NER labels
    :return: Most common NER label as string
    """
    counter = Counter(labels)
    return counter.most_common(1)[0][0]