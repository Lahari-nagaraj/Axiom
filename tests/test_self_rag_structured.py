"""Offline unit tests for the Self-RAG structured-output reliability fixes.

Uses only the standard-library ``unittest`` runner (no third-party test deps) so
they run in the existing project venv. All LLM interactions are mocked:
``src.self_rag._llm`` (or ``_structured_output``) is patched, so these tests make
NO network calls and consume no API quota. The real prompt formatting and node
logic are exercised.
"""

import unittest
from unittest import mock

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException

import src.self_rag as sr
from src.self_rag import (
    RelevanceDecision,
    RetrieveDecision,
    StructuredOutputError,
    decide_retrieval,
    grade_relevance,
)


class _FakeStructured:
    """Mimics the Runnable returned by ``ChatGroq.with_structured_output``."""

    def __init__(self, outcomes):
        # Each outcome is either a value to return or an Exception to raise.
        self._outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        outcome = self._outcomes[min(self.calls - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeLLM:
    def __init__(self, outcomes):
        self.structured = _FakeStructured(outcomes)

    def with_structured_output(self, schema, method=None, **kwargs):
        assert method == "json_schema", "expected the existing json_schema method"
        return self.structured


class LLMConfigTests(unittest.TestCase):
    def test_llm_sets_reasoning_format_parsed(self):
        class FakeSettings:
            groq_api_key = "test-key-not-used"
            groq_model = "openai/gpt-oss-120b"

        with mock.patch.object(sr, "get_settings", lambda: FakeSettings()):
            llm = sr._llm()
        # "parsed" keeps raw reasoning out of message.content.
        self.assertEqual(llm.reasoning_format, "parsed")
        self.assertEqual(llm.model_name, "openai/gpt-oss-120b")

    def test_llm_requires_api_key(self):
        class NoKey:
            groq_api_key = ""
            groq_model = "openai/gpt-oss-120b"

        with mock.patch.object(sr, "get_settings", lambda: NoKey()):
            with self.assertRaises(RuntimeError):
                sr._llm()


class DecideRetrievalTests(unittest.TestCase):
    def test_valid(self):
        fake = _FakeLLM([RetrieveDecision(should_retrieve=True)])
        with mock.patch.object(sr, "_llm", lambda: fake):
            result = decide_retrieval({"question": "payments CPU is above 85%"})

        self.assertTrue(result["need_retrieval"])
        self.assertEqual(fake.structured.calls, 1)
        self.assertTrue(any("Retrieval decision: True" in t for t in result["trace"]))

    def test_retries_malformed_then_succeeds(self):
        fake = _FakeLLM(
            [OutputParserException("schema echoed instead of decision"), RetrieveDecision(should_retrieve=False)]
        )
        with mock.patch.object(sr, "_llm", lambda: fake):
            result = decide_retrieval({"question": "what is a load balancer?"})

        self.assertFalse(result["need_retrieval"])
        self.assertEqual(fake.structured.calls, 2)  # bounded retry actually happened

    def test_exhausted_raises_without_fabricating(self):
        fake = _FakeLLM([OutputParserException("always bad")])
        with mock.patch.object(sr, "_llm", lambda: fake):
            with self.assertRaises(StructuredOutputError):
                decide_retrieval({"question": "anything"})
        # It did not silently skip retrieval or invent a decision.
        self.assertEqual(fake.structured.calls, sr.STRUCTURED_PARSE_ATTEMPTS)

    def test_bounded_attempts(self):
        fake = _FakeLLM([OutputParserException("bad")])
        with mock.patch.object(sr, "_llm", lambda: fake):
            with self.assertRaises(StructuredOutputError):
                sr._structured_output(RetrieveDecision, [], node="unit", attempts=3)
        self.assertEqual(fake.structured.calls, 3)

    def test_non_parse_errors_propagate(self):
        """Provider/quota errors must NOT be swallowed as parse failures."""
        fake = _FakeLLM([RuntimeError("429 rate limit / quota exceeded")])
        with mock.patch.object(sr, "_llm", lambda: fake):
            with self.assertRaises(RuntimeError):
                sr._structured_output(RetrieveDecision, [], node="unit")


class GradeRelevanceTests(unittest.TestCase):
    def test_skips_unparseable_doc(self):
        docs = [Document(page_content="first"), Document(page_content="second")]
        seq = {"n": 0}

        def fake_structured(model_cls, messages, *, node, attempts=None):
            seq["n"] += 1
            if seq["n"] == 1:
                raise StructuredOutputError("could not parse")
            return RelevanceDecision(is_relevant=True)

        with mock.patch.object(sr, "_structured_output", fake_structured):
            out = grade_relevance({"question": "q", "docs": docs, "source_mode": "internal"})

        self.assertEqual(len(out["relevant_docs"]), 1)
        self.assertTrue(any("ungraded" in t for t in out["trace"]))

    def test_propagates_provider_error(self):
        def fake_structured(*args, **kwargs):
            raise RuntimeError("429 rate limit / quota exceeded")

        with mock.patch.object(sr, "_structured_output", fake_structured):
            with self.assertRaises(RuntimeError):
                grade_relevance(
                    {"question": "q", "docs": [Document(page_content="a")], "source_mode": "internal"}
                )


class ParserRegressionTests(unittest.TestCase):
    def test_pydantic_parser_rejects_reasoning_that_echoes_schema(self):
        """Regression evidence: raw reasoning echoing the schema breaks parsing.

        This is why the client must set reasoning_format='parsed' so reasoning is
        kept out of message.content.
        """
        from langchain_core.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=RetrieveDecision)
        schema = (
            '{"title": "RetrieveDecision", "type": "object", "properties": '
            '{"should_retrieve": {"title": "Should Retrieve", "type": "boolean"}}, '
            '"required": ["should_retrieve"]}'
        )
        raw_reasoning = 'thinking the schema is ' + schema + ' \n{"should_retrieve": true}'

        with self.assertRaises(OutputParserException):
            parser.parse(raw_reasoning)

        # Clean content (what reasoning_format="parsed" produces) parses fine.
        self.assertTrue(parser.parse('{"should_retrieve": true}').should_retrieve)


if __name__ == "__main__":
    unittest.main()
