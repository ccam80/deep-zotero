"""Citation keys are read from Zotero's native citationKey field."""
import sqlite3
from pathlib import Path

from deep_zotero.zotero_client import ZoteroClient

# From conftest's mock library: itemID 1 is ABC12345 and itemID 2 is DEF67890,
# both with PDFs.
ITEM_ABC = 1
ITEM_DEF = 2


def _set_citation_keys(db_path: Path, keys: dict[int, str]) -> None:
    conn = sqlite3.connect(db_path)
    for value_id, (item_id, key) in enumerate(keys.items(), start=100):
        conn.execute(
            "INSERT INTO itemDataValues (valueID, value) VALUES (?, ?)",
            (value_id, key),
        )
        conn.execute(
            "INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, 9, ?)",
            (item_id, value_id),
        )
    conn.commit()
    conn.close()


def _items_by_key(db_path: Path) -> dict:
    return {i.item_key: i for i in ZoteroClient(db_path.parent).get_all_items_with_pdfs()}


class TestCitationKeys:
    def test_key_is_read_from_the_native_field(self, mock_zotero_db: Path):
        _set_citation_keys(mock_zotero_db, {ITEM_ABC: "smithStudy2024"})

        assert _items_by_key(mock_zotero_db)["ABC12345"].citation_key == "smithStudy2024"

    def test_item_without_a_key_is_blank_not_an_error(self, mock_zotero_db: Path):
        _set_citation_keys(mock_zotero_db, {ITEM_ABC: "smithStudy2024"})

        assert _items_by_key(mock_zotero_db)["DEF67890"].citation_key == ""

    def test_library_with_no_keys_at_all(self, mock_zotero_db: Path):
        items = _items_by_key(mock_zotero_db)

        assert {i.citation_key for i in items.values()} == {""}

    def test_keys_are_matched_to_the_right_items(self, mock_zotero_db: Path):
        _set_citation_keys(
            mock_zotero_db, {ITEM_ABC: "abcKey2024", ITEM_DEF: "defKey2019"}
        )

        items = _items_by_key(mock_zotero_db)

        assert items["ABC12345"].citation_key == "abcKey2024"
        assert items["DEF67890"].citation_key == "defKey2019"
