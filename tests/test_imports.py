def test_core_imports():
    from core.expression_layer import build_expression_plan, evaluate_expression_quality
    from core.query_orchestrator import route_query
    assert build_expression_plan is not None
    assert evaluate_expression_quality is not None
    assert route_query is not None


def test_memory_imports():
    from memory.background_memory import onboard_user_background, retrieve_user_background
    assert onboard_user_background is not None
    assert retrieve_user_background is not None


def test_retrieval_imports():
    from retrieval.rag_system import initialize_rag
    assert initialize_rag is not None
