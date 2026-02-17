from clueval.spans_table import BioToSentenceParser, BioToSpanParser

def test_bio_to_sentence(p1):
    pos_to_sent_mapping = BioToSentenceParser(p1)()
    assert list(pos_to_sent_mapping.keys()) == ["token_ids", "sents"]

    assert len(pos_to_sent_mapping["token_ids"]) == 8
    assert len(pos_to_sent_mapping["sents"]) == 8

    assert pos_to_sent_mapping["token_ids"][0][0] == 1
    assert pos_to_sent_mapping["token_ids"][0][-1] == 4

    assert pos_to_sent_mapping["sents"][0][0] == pos_to_sent_mapping["sents"][0][-1]

def test_bio_to_span_parser(p1):
    pos_to_span_mapping = BioToSpanParser(p1)()
    pass
