"""The ``write_metadata`` setting: what a permanent write carries beside its text.

Off by default, and off means the request is unchanged -- no metadata kwarg
value, no extra tool argument. On, every permanent write names when it was
written, by whom, from which Hermes session and by which lane, and
``cognee_remember`` accepts an optional flat ``metadata`` object that is
validated before anything is stored. ``cognee_recall`` hands stored metadata
back with each result. Run standalone with ``python3 tests/test_write_metadata.py``.
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


def _remember(provider, args):
    return json.loads(provider.handle_tool_call("cognee_remember", args))


def _remember_schema(provider):
    return next(s for s in provider.get_tool_schemas() if s["name"] == "cognee_remember")


class TestOff(unittest.TestCase):
    def test_a_tool_write_carries_no_metadata(self):
        with fake_backend() as fake:
            make_provider().handle_tool_call("cognee_remember", {"content": "fact"})
            self.assertIsNone(fake.only_call("remember_permanent")["metadata"])

    def test_the_tool_does_not_offer_metadata(self):
        provider = make_provider(config={"write_metadata": False})
        self.assertNotIn("metadata", _remember_schema(provider)["parameters"]["properties"])

    def test_metadata_passed_anyway_is_ignored_not_rejected(self):
        with fake_backend() as fake:
            envelope = _remember(make_provider(), {"content": "fact", "metadata": "junk"})
            self.assertNotIn("error", envelope)
            self.assertIsNone(fake.only_call("remember_permanent")["metadata"])


class TestOn(unittest.TestCase):
    def _metadata(self, args, **provider_kwargs):
        with fake_backend() as fake:
            provider = make_provider(write_metadata=True, **provider_kwargs)
            envelope = _remember(provider, args)
            calls = fake.kwargs_for("remember_permanent")
        return envelope, (calls[0]["metadata"] if calls else None), len(calls)

    def test_a_tool_write_names_its_time_writer_session_and_lane(self):
        before = datetime.now(timezone.utc) - timedelta(seconds=1)
        _, meta, _ = self._metadata({"content": "fact"}, session_id="sess-42")
        self.assertEqual(meta["write_origin"], "cognee_remember")
        self.assertEqual(meta["created_by"], "hermes")
        # The Hermes session id, not the cognee one, so it points at the transcript.
        self.assertEqual(meta["hermes_session_id"], "sess-42")
        created = datetime.fromisoformat(meta["created_at"])
        self.assertEqual(created.utcoffset(), timedelta(0))
        self.assertGreaterEqual(created, before.replace(microsecond=0))
        self.assertEqual(
            set(meta), {"created_at", "created_by", "write_origin", "hermes_session_id"}
        )

    def test_created_by_follows_the_setting(self):
        with fake_backend() as fake:
            provider = make_provider(write_metadata=True)
            provider._created_by = "hermes-dinefile"
            provider.handle_tool_call("cognee_remember", {"content": "fact"})
            self.assertEqual(
                fake.only_call("remember_permanent")["metadata"]["created_by"], "hermes-dinefile"
            )

    def test_the_automatic_keys_cannot_be_overwritten(self):
        _, meta, _ = self._metadata(
            {
                "content": "fact",
                "metadata": {"created_by": "claude", "write_origin": "x", "hermes_session_id": "y"},
            },
            session_id="sess-1",
        )
        self.assertEqual(meta["created_by"], "hermes")
        self.assertEqual(meta["write_origin"], "cognee_remember")
        self.assertEqual(meta["hermes_session_id"], "sess-1")

    def test_no_session_id_means_no_session_key(self):
        _, meta, _ = self._metadata({"content": "fact"}, session_id="")
        self.assertNotIn("hermes_session_id", meta)

    def test_the_tool_offers_a_metadata_object(self):
        provider = make_provider(write_metadata=True, config={"write_metadata": True})
        properties = _remember_schema(provider)["parameters"]["properties"]
        self.assertEqual(properties["metadata"]["type"], "object")
        self.assertEqual(set(properties), {"content", "dataset", "metadata"})

    def test_caller_metadata_is_stored_beside_the_automatic_keys(self):
        _, meta, _ = self._metadata(
            {
                "content": "fact",
                "metadata": {
                    "source_id": "ticket-42",
                    "priority": 2,
                    "verified": True,
                    "gone": None,
                },
            }
        )
        self.assertEqual(meta["source_id"], "ticket-42")
        self.assertEqual(meta["priority"], 2)
        self.assertIs(meta["verified"], True)
        # A None value is dropped, not stored and not an error.
        self.assertNotIn("gone", meta)
        self.assertEqual(meta["created_by"], "hermes")

    def test_invalid_metadata_stores_nothing(self):
        cases = {
            "not an object": "a string",
            "nested value": {"k": {"inner": 1}},
            "list value": {"k": [1, 2]},
            "reserved node_set": {"node_set": "x"},
            "cognee namespace": {"_cognee": "x"},
            "empty key": {" ": "x"},
            "long key": {"k" * 65: "x"},
            "long value": {"k": "v" * 501},
            "too many keys": {f"k{i}": i for i in range(17)},
        }
        for label, metadata in cases.items():
            with self.subTest(label):
                envelope, _, calls = self._metadata({"content": "fact", "metadata": metadata})
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
        self.assertEqual(meta["created_by"], "hermes")
        self.assertEqual(meta["hermes_session_id"], "sess-7")

    def test_the_callers_write_origin_is_kept(self):
        meta = self._mirror_metadata(
            "add", "user", "likes tea", metadata={"write_origin": "background_review"}
        )
        self.assertEqual(meta["write_origin"], "background_review")


class TestRecallShowsMetadata(unittest.TestCase):
    # The shape a CHUNKS row has over HTTP on cognee 1.6.1, observed live: the
    # chunk payload flat, external_metadata as JSON text beside the score.
    _STORED = {
        "created_at": "2026-09-29T16:00:00+00:00",
        "created_by": "hermes",
        "source_id": "ticket-42",
        "_cognee": {"source_uri": "file:///app/memory-x.txt"},
    }

    def _recall(self, row):
        with fake_backend() as fake:
            fake.results["recall"] = [row]
            return json.loads(
                make_provider().handle_tool_call("cognee_recall", {"query": "morning workouts"})
            )

    def test_a_chunks_row_carries_its_metadata_to_the_agent(self):
        row = {
            "text": "prefers mornings",
            "score": 0.1,
            "external_metadata": json.dumps(self._STORED),
        }
        item = self._recall(row)["results"][0]
        self.assertEqual(item["text"], "prefers mornings")
        self.assertEqual(
            item["metadata"],
            {
                "created_at": "2026-09-29T16:00:00+00:00",
                "created_by": "hermes",
                "source_id": "ticket-42",
            },
        )

    def test_a_nested_payload_is_read_too(self):
        row = {"text": "t", "payload": {"external_metadata": {"created_by": "hermes"}}}
        self.assertEqual(self._recall(row)["results"][0]["metadata"], {"created_by": "hermes"})

    def test_a_row_without_usable_metadata_is_unchanged(self):
        for raw in (None, "", "not json", "[1, 2]", json.dumps({"_cognee": {}})):
            with self.subTest(raw=raw):
                item = self._recall({"text": "t", "external_metadata": raw})["results"][0]
                self.assertNotIn("metadata", item)


if __name__ == "__main__":
    unittest.main()
