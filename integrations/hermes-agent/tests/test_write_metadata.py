"""The ``write_metadata`` setting: what a permanent write carries beside its text.

Off by default, and off means the request is unchanged -- no metadata kwarg
value, no extra tool argument. On, every permanent write names when it was
written, from which Hermes session and by which lane, and ``cognee_remember``
accepts an optional Notion page pointer that is validated before anything is
stored. Run standalone with ``python3 tests/test_write_metadata.py``.
"""

import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import fake_backend, make_provider  # noqa: E402
from cognee_integration_hermes.provider import _normalize_notion_page_id  # noqa: E402

_PAGE = "1a2b3c4d5e6f47a8b9c0d1e2f3a4b5c6"
_PAGE_DASHED = "1a2b3c4d-5e6f-47a8-b9c0-d1e2f3a4b5c6"


def _remember(provider, args):
    return json.loads(provider.handle_tool_call("cognee_remember", args))


def _remember_schema(provider):
    return next(s for s in provider.get_tool_schemas() if s["name"] == "cognee_remember")


class TestOff(unittest.TestCase):
    def test_a_tool_write_carries_no_metadata(self):
        with fake_backend() as fake:
            make_provider().handle_tool_call("cognee_remember", {"content": "fact"})
            self.assertIsNone(fake.only_call("remember_permanent")["metadata"])

    def test_the_tool_does_not_offer_a_notion_page_id(self):
        provider = make_provider(config={"write_metadata": False})
        self.assertNotIn("notion_page_id", _remember_schema(provider)["parameters"]["properties"])

    def test_a_notion_page_id_passed_anyway_is_ignored_not_rejected(self):
        with fake_backend() as fake:
            envelope = _remember(make_provider(), {"content": "fact", "notion_page_id": "junk"})
            self.assertNotIn("error", envelope)
            self.assertIsNone(fake.only_call("remember_permanent")["metadata"])


class TestOn(unittest.TestCase):
    def _metadata(self, args, **provider_kwargs):
        with fake_backend() as fake:
            provider = make_provider(write_metadata=True, **provider_kwargs)
            envelope = _remember(provider, args)
            calls = fake.kwargs_for("remember_permanent")
        return envelope, (calls[0]["metadata"] if calls else None), len(calls)

    def test_a_tool_write_names_its_time_session_and_lane(self):
        before = datetime.now(timezone.utc) - timedelta(seconds=1)
        _, meta, _ = self._metadata({"content": "fact"}, session_id="sess-42")
        self.assertEqual(meta["write_origin"], "cognee_remember")
        # The Hermes session id, not the cognee one, so it points at the transcript.
        self.assertEqual(meta["hermes_session_id"], "sess-42")
        created = datetime.fromisoformat(meta["created_at"])
        self.assertEqual(created.utcoffset(), timedelta(0))
        self.assertGreaterEqual(created, before.replace(microsecond=0))
        self.assertNotIn("notion_page_id", meta)

    def test_no_session_id_means_no_session_key(self):
        _, meta, _ = self._metadata({"content": "fact"}, session_id="")
        self.assertNotIn("hermes_session_id", meta)

    def test_the_tool_offers_a_notion_page_id(self):
        provider = make_provider(write_metadata=True, config={"write_metadata": True})
        properties = _remember_schema(provider)["parameters"]["properties"]
        self.assertIn("notion_page_id", properties)
        self.assertEqual(set(properties) - {"notion_page_id"}, {"content", "dataset"})

    def test_a_valid_notion_page_id_is_stored_canonical(self):
        _, meta, _ = self._metadata({"content": "fact", "notion_page_id": _PAGE})
        self.assertEqual(meta["notion_page_id"], _PAGE_DASHED)

    def test_an_invalid_notion_page_id_stores_nothing(self):
        # The truncated-URL bug: a pointer that resolves to nothing reads as usable.
        envelope, _, calls = self._metadata(
            {"content": "fact", "notion_page_id": "https://app.notion.so"}
        )
        self.assertIn("notion_page_id", envelope["error"])
        self.assertIn("Nothing was stored", envelope["error"])
        self.assertEqual(calls, 0)


class TestMemoryWriteMirror(unittest.TestCase):
    def _mirror_metadata(self, *args, **kwargs):
        with fake_backend() as fake:
            provider = make_provider(write_metadata=True, session_id="sess-7")
            provider.on_memory_write(*args, **kwargs)
            self.assertTrue(fake.wait("remember_permanent", timeout=2.0))
            return fake.only_call("remember_permanent")["metadata"]

    def test_the_default_origin_is_the_memory_tool(self):
        meta = self._mirror_metadata("add", "project", "prefers tabs")
        self.assertEqual(meta["write_origin"], "hermes_memory_tool")
        self.assertEqual(meta["hermes_session_id"], "sess-7")

    def test_the_callers_write_origin_is_kept(self):
        meta = self._mirror_metadata(
            "add", "user", "likes tea", metadata={"write_origin": "background_review"}
        )
        self.assertEqual(meta["write_origin"], "background_review")


class TestNotionPageId(unittest.TestCase):
    def test_accepted_forms(self):
        cases = {
            "bare": _PAGE,
            "dashed": _PAGE_DASHED,
            "upper case": _PAGE.upper(),
            "padded": f"  {_PAGE}  ",
            "url": f"https://www.notion.so/workspace/Project-Plan-{_PAGE}",
            # A hex-looking last title word must not be glued onto the id.
            "hex title word": f"https://www.notion.so/Cafe-Bad-{_PAGE}",
            "url with a view": f"https://www.notion.so/{_PAGE}?v=ffffffffffffffffffffffffffffffff",
            "fragment": f"https://www.notion.so/Plan-{_PAGE}#abc",
        }
        for label, value in cases.items():
            with self.subTest(label):
                self.assertEqual(_normalize_notion_page_id(value), _PAGE_DASHED)

    def test_a_peek_url_names_the_page_in_p_not_the_database_in_the_path(self):
        database = "ffffffffffffffffffffffffffffffff"
        url = f"https://www.notion.so/{database}?v=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee&p={_PAGE}"
        self.assertEqual(_normalize_notion_page_id(url), _PAGE_DASHED)

    def test_rejected_forms(self):
        for value in (
            "",
            None,
            "https://app.notion.so",
            _PAGE[:31],
            _PAGE + "a",
            "not a page",
        ):
            with self.subTest(value=value):
                self.assertEqual(_normalize_notion_page_id(value), "")


if __name__ == "__main__":
    unittest.main()
