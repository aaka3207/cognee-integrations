"""Fork-only: the ``notion_page_id`` metadata convention (notion_pointer.py).

Run standalone with ``python3 tests/test_notion_pointer.py``.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import fake_backend, make_provider  # noqa: E402
from cognee_integration_hermes.notion_pointer import normalize_page_id  # noqa: E402

_PAGE = "1a2b3c4d5e6f47a8b9c0d1e2f3a4b5c6"
_PAGE_DASHED = "1a2b3c4d-5e6f-47a8-b9c0-d1e2f3a4b5c6"


class TestThroughTheTool(unittest.TestCase):
    def _write(self, metadata):
        with fake_backend() as fake:
            provider = make_provider(write_metadata=True)
            envelope = json.loads(
                provider.handle_tool_call(
                    "cognee_remember", {"content": "fact", "metadata": metadata}
                )
            )
            calls = fake.kwargs_for("remember_permanent")
        return envelope, (calls[0]["metadata"] if calls else None), len(calls)

    def test_a_valid_page_id_is_stored_canonical(self):
        _, meta, _ = self._write({"notion_page_id": _PAGE, "other": "kept"})
        self.assertEqual(meta["notion_page_id"], _PAGE_DASHED)
        self.assertEqual(meta["other"], "kept")

    def test_an_invalid_page_id_stores_nothing(self):
        # The truncated-URL bug: a pointer that resolves to nothing reads as usable.
        envelope, _, calls = self._write({"notion_page_id": "https://app.notion.so"})
        self.assertIn("notion_page_id", envelope["error"])
        self.assertIn("Nothing was stored", envelope["error"])
        self.assertEqual(calls, 0)

    def test_metadata_without_the_key_is_untouched(self):
        _, meta, _ = self._write({"source_id": "x"})
        self.assertNotIn("notion_page_id", meta)


class TestNormalizePageId(unittest.TestCase):
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
                self.assertEqual(normalize_page_id(value), _PAGE_DASHED)

    def test_a_peek_url_names_the_page_in_p_not_the_database_in_the_path(self):
        database = "ffffffffffffffffffffffffffffffff"
        url = f"https://www.notion.so/{database}?v=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee&p={_PAGE}"
        self.assertEqual(normalize_page_id(url), _PAGE_DASHED)

    def test_rejected_forms(self):
        for value in ("", None, "https://app.notion.so", _PAGE[:31], _PAGE + "a", "not a page"):
            with self.subTest(value=value):
                self.assertEqual(normalize_page_id(value), "")


if __name__ == "__main__":
    unittest.main()
