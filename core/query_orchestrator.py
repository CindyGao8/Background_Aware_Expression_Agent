import os
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
@dataclass
class QueryUnderstanding:
    """Structured understanding of the user's query.

    This object is intentionally explicit so downstream modules do not need to
    infer intent from raw text again.
    """

    query_id: str = "q_auto"
    user_id: str = "default_user"
    raw_query: str = ""
    normalized_query: str = ""
    query_type: str = "concept_explanation"
    topic: str = ""
    subtopics: List[str] = field(default_factory=list)
    intent: str = "explain"
    domain: str = ""
    target_audience_hint: Optional[str] = None
    expected_answer_scope: str = "general"
    requires_background_retrieval: bool = True
    requires_project_context: bool = False
    requires_external_knowledge: bool = False
    needs_clarification: bool = False
    clarification_reason: str = ""
    suggested_clarification_question: str = ""
    recommended_background_chunk_types: List[str] = field(default_factory=list)
    recommended_next_step: str = "retrieve_background_then_explain"
    confidence: float = 0.75
    routing_rationale: str = ""
    risk_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoutingDecision:
    """Routing decision for the downstream explanation pipeline."""

    route: str
    message: str = ""
    background_request: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.75
    fallback_route: Optional[str] = None
    risk_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


VALID_QUERY_TYPES = {
    "concept_explanation",
    "project_explanation",
    "comparison_question",
    "workflow_explanation",
    "document_based_question",
    "clarification_needed",
}

VALID_NEXT_STEPS = {
    "clarification",
    "retrieve_background_then_explain",
    "retrieve_background_and_project_then_explain",
    "external_knowledge_then_explain",
}

VALID_CHUNK_TYPES = {
    "role_identity",
    "domain_context",
    "technical_exposure",
    "knowledge_boundary",
    "expression_preference",
    "current_project",
}

AUDIENCE_ALIASES = {
    "engineer": "engineer",
    "developer": "engineer",
    "technical": "engineer",
    "pm": "product_manager",
    "product manager": "product_manager",
    "business": "business_owner",
    "business owner": "business_owner",
    "executive": "business_owner",
    "general": "general",
}
from openai import OpenAI


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def _parse_json_safely(text: str) -> Dict:
    if not text:
        raise ValueError("Empty response from model.")

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    return json.loads(text)


def _normalize_query(raw_query: str) -> str:
    return " ".join((raw_query or "").strip().split())


def _infer_audience_hint(raw_query: str) -> Optional[str]:
    q = (raw_query or "").lower()
    for phrase, audience in AUDIENCE_ALIASES.items():
        if f"for {phrase}" in q or f"to a {phrase}" in q or f"as an {phrase}" in q or f"as a {phrase}" in q:
            return audience
    return None


def _repair_common_typos(raw_query: str) -> str:
    repaired = _normalize_query(raw_query)
    replacements = {
        "what:s": "what is",
        "whats ": "what is ",
        "what's": "what is",
        "explainn": "explain",
        "retreival": "retrieval",
        "retrival": "retrieval",
        "embeding": "embedding",
    }
    lowered = repaired.lower()
    for wrong, right in replacements.items():
        if wrong in lowered:
            repaired = repaired.replace(wrong, right).replace(wrong.title(), right.title())
            lowered = repaired.lower()
    return repaired


def _looks_like_minor_typo(raw_query: str, repaired_query: str) -> bool:
    return _normalize_query(raw_query).lower() != _normalize_query(repaired_query).lower()


def _detect_query_risks(raw_query: str, result: Dict[str, Any]) -> List[str]:
    risks: List[str] = []
    q = (raw_query or "").strip()
    if len(q) < 4:
        risks.append("Query is very short and may be ambiguous.")
    if result.get("needs_clarification"):
        risks.append("Model marked the query as needing clarification.")
    if result.get("requires_project_context") and not result.get("requires_external_knowledge"):
        risks.append("Answer depends on project context; missing or weak project retrieval may reduce answer quality.")
    if result.get("requires_external_knowledge") and result.get("requires_project_context"):
        risks.append("Query may require both external knowledge and project-specific context.")
    return risks


def _coerce_confidence(value: Any, default: float = 0.75) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return round(max(0.0, min(1.0, score)), 2)


def _is_potentially_ambiguous_query(raw_query: str) -> Dict:
    q = raw_query.strip().lower().replace("?", "")

    ambiguous_terms = {
        "rag": {
            "topic": "RAG",
            "question": (
                'Do you mean "Retrieval-Augmented Generation" in AI, '
                'or "Red-Amber-Green" in project/status reporting?'
            )
        },
        "orchestrator": {
            "topic": "orchestrator",
            "question": (
                "Do you mean an orchestrator in AI/agent systems, "
                "or a project-specific orchestrator in your uploaded documents?"
            )
        },
        "api gateway": {
            "topic": "API gateway",
            "question": (
                "Do you want a general software architecture explanation of API Gateway, "
                "or are you referring to a specific project component?"
            )
        },
    }

    for term, meta in ambiguous_terms.items():
        if q == term or q == f"what is {term}":
            return {
                "is_ambiguous": True,
                "topic": meta["topic"],
                "question": meta["question"]
            }

    return {
        "is_ambiguous": False,
        "topic": None,
        "question": None
    }


def _default_background_chunk_types(query_type: str) -> List[str]:
    if query_type == "concept_explanation":
        return [
            "technical_exposure",
            "knowledge_boundary",
            "expression_preference"
        ]
    if query_type == "comparison_question":
        return [
            "technical_exposure",
            "knowledge_boundary",
            "expression_preference",
            "role_identity"
        ]
    if query_type in {"project_explanation", "document_based_question"}:
        return [
            "current_project",
            "role_identity",
            "expression_preference"
        ]
    if query_type == "workflow_explanation":
        return [
            "knowledge_boundary",
            "expression_preference",
            "role_identity",
            "current_project"
        ]
    return [
        "role_identity",
        "technical_exposure",
        "expression_preference"
    ]


def _smart_chunk_selection(query_type: str, topic: str, intent: str, domain: str) -> List[str]:
    """
    Smarter chunk selection for Person 2.
    This is the main fix for problem 1.
    """
    qt = (query_type or "").lower()
    tp = (topic or "").lower()
    it = (intent or "").lower()
    dm = (domain or "").lower()

    # Project / uploaded-doc style questions
    if qt in {"project_explanation", "document_based_question"}:
        return [
            "current_project",
            "role_identity",
            "expression_preference"
        ]

    # Workflow / architecture style questions
    if qt == "workflow_explanation":
        return [
            "knowledge_boundary",
            "expression_preference",
            "technical_exposure",
            "role_identity"
        ]

    # Comparison questions
    if qt == "comparison_question":
        return [
            "technical_exposure",
            "knowledge_boundary",
            "expression_preference",
            "role_identity"
        ]

    # Concept explanations
    if qt == "concept_explanation":
        # Orchestrator / architecture-like concept
        if any(x in tp for x in ["orchestrator", "system", "architecture", "agent"]) or \
           any(x in it for x in ["role", "function", "workflow"]) or \
           "artificial intelligence" in dm or "ai" in dm:
            return [
                "knowledge_boundary",
                "expression_preference",
                "technical_exposure"
            ]

        # Definitions like RAG / vector DB / API gateway
        if any(x in tp for x in ["rag", "retrieval", "vector", "database", "api", "gateway"]):
            return [
                "technical_exposure",
                "expression_preference",
                "knowledge_boundary"
            ]

        # Generic concept fallback
        return [
            "technical_exposure",
            "expression_preference",
            "domain_context"
        ]

    return _default_background_chunk_types(qt)


def understand_query(
    user_id: str,
    raw_query: str,
    has_uploaded_project_doc: bool = False
) -> Dict:
    """
    Person 2 - Query Understanding
    Produces a Query Understanding Object.
    """

    raw_query = raw_query.strip()

    normalized_query = _repair_common_typos(raw_query)
    target_audience_hint = _infer_audience_hint(normalized_query)
    typo_was_repaired = _looks_like_minor_typo(raw_query, normalized_query)

    # -----------------------------
    # Rule-based ambiguity first pass
    # -----------------------------
    ambiguity = _is_potentially_ambiguous_query(raw_query)
    if ambiguity["is_ambiguous"]:
        return {
            "query_id": "q_auto",
            "user_id": user_id,
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "query_type": "clarification_needed",
            "topic": ambiguity["topic"],
            "subtopics": [],
            "intent": "resolve_ambiguity",
            "domain": "",
            "target_audience_hint": target_audience_hint,
            "expected_answer_scope": "clarification",
            "requires_background_retrieval": False,
            "requires_project_context": False,
            "requires_external_knowledge": False,
            "needs_clarification": True,
            "clarification_reason": "The query term is ambiguous and has multiple plausible meanings.",
            "suggested_clarification_question": ambiguity["question"],
            "recommended_background_chunk_types": [],
            "confidence": 0.95,
            "routing_rationale": "Rule-based ambiguity detection matched a known ambiguous term.",
            "risk_flags": ["Ambiguous short query requires clarification."],
            "recommended_next_step": "clarification",
        }

    client = _get_openai_client()

    prompt = f"""
You are the query understanding module of a personalized explanation agent.

Your job is to classify the user's query and decide what type of context is needed.

Return ONLY valid JSON with exactly these keys:

- query_id
- user_id
- raw_query
- normalized_query
- query_type
- topic
- subtopics
- intent
- domain
- target_audience_hint
- expected_answer_scope
- requires_background_retrieval
- requires_project_context
- requires_external_knowledge
- needs_clarification
- clarification_reason
- suggested_clarification_question
- recommended_background_chunk_types
- recommended_next_step
- confidence
- routing_rationale
- risk_flags

Allowed values:
- query_type must be one of:
  ["concept_explanation", "project_explanation", "comparison_question", "workflow_explanation", "document_based_question", "clarification_needed"]

- recommended_background_chunk_types must be chosen from:
  ["role_identity", "domain_context", "technical_exposure", "knowledge_boundary", "expression_preference", "current_project"]

- recommended_next_step must be one of:
  ["clarification", "retrieve_background_then_explain", "retrieve_background_and_project_then_explain", "external_knowledge_then_explain"]

- expected_answer_scope should be one of:
  ["definition", "implementation", "workflow", "comparison", "project_specific", "decision_support", "clarification", "general"]

- target_audience_hint should be one of:
  ["engineer", "product_manager", "business_owner", "general", null]

Important routing rules:
1. If the user asks a general concept question like:
   - What is retrieval-augmented generation?
   - What is a vector database?
   - Explain API gateway
   - Explain what an orchestrator does in an AI agent system
   then this is usually:
   - query_type = "concept_explanation"
   - requires_external_knowledge = true
   - requires_project_context = false

2. Only set requires_project_context = true when the question clearly depends on uploaded project documents, such as:
   - Explain this project
   - Explain this architecture in the uploaded note
   - What is the study design in the uploaded document?
   - Summarize the uploaded note
   - What does this project document say about X?

3. If the question is vague and depends on missing context, set:
   - needs_clarification = true

4. Background retrieval is usually useful for personalization.
5. Do not route to clarification for minor typos when the intended query is obvious. Use normalized_query to repair small typos and continue.
6. If the user explicitly asks for an engineer, PM/product manager, business owner, executive, or general-user explanation, set target_audience_hint accordingly.
7. Include a short routing_rationale explaining why the selected next step is appropriate.
8. Set confidence between 0 and 1.

Current context:
- user_id = "{user_id}"
- has_uploaded_project_doc = {str(has_uploaded_project_doc)}
- raw_query = "{raw_query}"
- normalized_query = "{normalized_query}"
- target_audience_hint = "{target_audience_hint}"
- minor_typo_repaired = {str(typo_was_repaired)}
"""

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.5"),
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    result = _parse_json_safely(content)

    if result.get("query_type") not in VALID_QUERY_TYPES:
        result["query_type"] = "concept_explanation"

    if not isinstance(result.get("subtopics"), list):
        result["subtopics"] = []

    for key in [
        "requires_background_retrieval",
        "requires_project_context",
        "requires_external_knowledge",
        "needs_clarification",
    ]:
        if not isinstance(result.get(key), bool):
            result[key] = False

    # -----------------------------
    # SMART FIX FOR PROBLEM 1
    # Override chunk selection with smarter logic
    # -----------------------------
    smart_chunks = _smart_chunk_selection(
        query_type=result.get("query_type", ""),
        topic=result.get("topic", ""),
        intent=result.get("intent", ""),
        domain=result.get("domain", "")
    )

    result["recommended_background_chunk_types"] = [
        x for x in smart_chunks if x in VALID_CHUNK_TYPES
    ]

    if not result["recommended_background_chunk_types"]:
        result["recommended_background_chunk_types"] = _default_background_chunk_types(
            result.get("query_type", "concept_explanation")
        )

    if result.get("recommended_next_step") not in VALID_NEXT_STEPS:
        if result.get("needs_clarification", False):
            result["recommended_next_step"] = "clarification"
        elif result.get("requires_external_knowledge", False):
            result["recommended_next_step"] = "external_knowledge_then_explain"
        elif result.get("requires_project_context", False):
            result["recommended_next_step"] = "retrieve_background_and_project_then_explain"
        else:
            result["recommended_next_step"] = "retrieve_background_then_explain"

    result["user_id"] = user_id
    result["raw_query"] = raw_query

    if not result.get("query_id"):
        result["query_id"] = "q_auto"

    result["normalized_query"] = result.get("normalized_query") or normalized_query
    if typo_was_repaired and not result.get("needs_clarification", False):
        result["raw_query"] = raw_query
        result["normalized_query"] = normalized_query

    result["target_audience_hint"] = result.get("target_audience_hint") or target_audience_hint
    if result.get("target_audience_hint") not in {"engineer", "product_manager", "business_owner", "general", None}:
        result["target_audience_hint"] = target_audience_hint

    if result.get("expected_answer_scope") not in {
        "definition", "implementation", "workflow", "comparison", "project_specific", "decision_support", "clarification", "general"
    }:
        result["expected_answer_scope"] = "general"

    result["confidence"] = _coerce_confidence(result.get("confidence"), default=0.75)
    if not isinstance(result.get("routing_rationale"), str):
        result["routing_rationale"] = ""
    if not isinstance(result.get("risk_flags"), list):
        result["risk_flags"] = []

    inferred_risks = _detect_query_risks(raw_query, result)
    if typo_was_repaired:
        inferred_risks.append(f"Minor typo repaired: '{raw_query}' -> '{normalized_query}'")
    result["risk_flags"] = list(dict.fromkeys(result.get("risk_flags", []) + inferred_risks))

    understanding = QueryUnderstanding(
        query_id=result.get("query_id", "q_auto"),
        user_id=result.get("user_id", user_id),
        raw_query=raw_query,
        normalized_query=result.get("normalized_query", normalized_query),
        query_type=result.get("query_type", "concept_explanation"),
        topic=result.get("topic", ""),
        subtopics=result.get("subtopics", []),
        intent=result.get("intent", "explain"),
        domain=result.get("domain", ""),
        target_audience_hint=result.get("target_audience_hint"),
        expected_answer_scope=result.get("expected_answer_scope", "general"),
        requires_background_retrieval=result.get("requires_background_retrieval", True),
        requires_project_context=result.get("requires_project_context", False),
        requires_external_knowledge=result.get("requires_external_knowledge", False),
        needs_clarification=result.get("needs_clarification", False),
        clarification_reason=result.get("clarification_reason", ""),
        suggested_clarification_question=result.get("suggested_clarification_question", ""),
        recommended_background_chunk_types=result.get("recommended_background_chunk_types", []),
        recommended_next_step=result.get("recommended_next_step", "retrieve_background_then_explain"),
        confidence=result.get("confidence", 0.75),
        routing_rationale=result.get("routing_rationale", ""),
        risk_flags=result.get("risk_flags", []),
    )

    return understanding.to_dict()


def route_query(query_understanding_object: Dict) -> Dict:
    """
    Person 2 - Routing decision.

    Converts the query-understanding object into an explicit route for the
    downstream pipeline.
    """

    base_request = {
        "user_id": query_understanding_object["user_id"],
        "query": query_understanding_object.get("normalized_query") or query_understanding_object["raw_query"],
        "raw_query": query_understanding_object["raw_query"],
        "recommended_background_chunk_types": query_understanding_object.get(
            "recommended_background_chunk_types", []
        ),
        "target_audience_hint": query_understanding_object.get("target_audience_hint"),
        "expected_answer_scope": query_understanding_object.get("expected_answer_scope"),
    }

    confidence = _coerce_confidence(query_understanding_object.get("confidence"), default=0.75)
    risk_flags = list(query_understanding_object.get("risk_flags", []))

    if query_understanding_object.get("needs_clarification", False):
        decision = RoutingDecision(
            route="clarification",
            message=query_understanding_object.get(
                "suggested_clarification_question",
                "Could you clarify what kind of explanation you want?"
            ),
            background_request=base_request,
            rationale=query_understanding_object.get(
                "routing_rationale",
                "The query understanding module determined that user clarification is needed."
            ),
            confidence=confidence,
            fallback_route="background_retrieval_then_expression",
            risk_flags=risk_flags,
        )
        return decision.to_dict()

    if query_understanding_object.get("requires_external_knowledge", False):
        decision = RoutingDecision(
            route="external_knowledge_then_expression",
            background_request=base_request,
            rationale=query_understanding_object.get(
                "routing_rationale",
                "The query is a general knowledge question that benefits from external knowledge before personalization."
            ),
            confidence=confidence,
            fallback_route="background_retrieval_then_expression",
            risk_flags=risk_flags,
        )
        return decision.to_dict()

    if query_understanding_object.get("requires_project_context", False):
        decision = RoutingDecision(
            route="background_and_project_then_expression",
            background_request=base_request,
            rationale=query_understanding_object.get(
                "routing_rationale",
                "The query depends on project-specific context and user background."
            ),
            confidence=confidence,
            fallback_route="background_retrieval_then_expression",
            risk_flags=risk_flags,
        )
        return decision.to_dict()

    decision = RoutingDecision(
        route="background_retrieval_then_expression",
        background_request=base_request,
        rationale=query_understanding_object.get(
            "routing_rationale",
            "The query can be answered with background-personalized explanation."
        ),
        confidence=confidence,
        fallback_route=None,
        risk_flags=risk_flags,
    )
    return decision.to_dict()


def process_query(
    user_id: str,
    raw_query: str,
    has_uploaded_project_doc: bool = False
) -> Dict:
    """
    One-step wrapper:
    returns both query understanding and routing decision.
    """
    q_obj = understand_query(
        user_id=user_id,
        raw_query=_repair_common_typos(raw_query),
        has_uploaded_project_doc=has_uploaded_project_doc
    )

    routing = route_query(q_obj)

    return {
        "query_understanding_object": q_obj,
        "routing_decision": routing,
        "orchestration_metadata": {
            "module": "query_orchestrator",
            "normalized_query": q_obj.get("normalized_query"),
            "target_audience_hint": q_obj.get("target_audience_hint"),
            "expected_answer_scope": q_obj.get("expected_answer_scope"),
            "confidence": q_obj.get("confidence"),
            "risk_flags": q_obj.get("risk_flags", []),
        }
    }
