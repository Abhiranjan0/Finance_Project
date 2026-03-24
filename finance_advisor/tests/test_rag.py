from finance_advisor.backend.rag.retriever import retrieve_top_k


def test_rag():
    chunks = retrieve_top_k("equity", 3)
    assert isinstance(chunks, list)
