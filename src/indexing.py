from .search_engine import SearchEngine


def build_index() -> SearchEngine:
    """
    Helper to create a SearchEngine instance and index all documents.
    """
    engine = SearchEngine()
    engine.index_corpus()
    return engine
