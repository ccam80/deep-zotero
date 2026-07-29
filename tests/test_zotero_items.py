"""Which Zotero items count as indexable sources.

Item type IDs come from Zotero's globalSchema and differ between versions, so
conftest's mock deliberately numbers them differently from a real library: any
code that hardcodes the IDs fails here.
"""
import sqlite3
from pathlib import Path

from deep_zotero.zotero_client import ZoteroClient

ATTACHMENT, NOTE = 7, 8


def _add_item(db_path: Path, item_id: int, item_key: str, type_id: int) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO items (itemID, itemTypeID, key) VALUES (?, ?, ?)",
        (item_id, type_id, item_key),
    )
    conn.commit()
    conn.close()


def _add_pdf(db_path: Path, item_id: int, parent_id: int | None) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO itemAttachments "
        "(itemID, parentItemID, path, contentType, linkMode) "
        "VALUES (?, ?, ?, 'application/pdf', 0)",
        (item_id, parent_id, f"storage:{item_id}.pdf"),
    )
    conn.commit()
    conn.close()


def _item_keys(db_path: Path) -> set[str]:
    return {i.item_key for i in ZoteroClient(db_path.parent).get_all_items_with_pdfs()}


class TestItemTypeFiltering:
    def test_parent_items_with_pdfs_are_returned(self, mock_zotero_db: Path):
        assert _item_keys(mock_zotero_db) == {"ABC12345", "DEF67890"}

    def test_standalone_pdf_attachment_is_not_a_source(self, mock_zotero_db: Path):
        """A PDF dragged in with no parent item is a file, not a citable source."""
        _add_item(mock_zotero_db, 200, "LOOSE_PDF", ATTACHMENT)
        _add_pdf(mock_zotero_db, 200, None)

        assert "LOOSE_PDF" not in _item_keys(mock_zotero_db)

    def test_child_attachment_is_not_returned_in_its_own_right(
        self, mock_zotero_db: Path
    ):
        """The attachment rows already in the fixture must not appear as items."""
        keys = _item_keys(mock_zotero_db)

        assert "ATT00001" not in keys
        assert "ATT00002" not in keys

    def test_parent_keeps_its_pdf_when_attachments_are_excluded(
        self, mock_zotero_db: Path
    ):
        """Excluding attachment-type items must not cost a source its PDF."""
        data_dir = mock_zotero_db.parent
        pdf = data_dir / "storage" / "ATT00001" / "test1.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b"%PDF-1.4\n")

        items = {
            i.item_key: i for i in ZoteroClient(data_dir).get_all_items_with_pdfs()
        }

        assert items["ABC12345"].title == "Heart Rate Paper"
        assert items["ABC12345"].pdf_path == pdf

    def test_notes_are_not_sources(self, mock_zotero_db: Path):
        _add_item(mock_zotero_db, 201, "A_NOTE", NOTE)
        _add_pdf(mock_zotero_db, 201, None)

        assert "A_NOTE" not in _item_keys(mock_zotero_db)

    def test_unusual_bibliographic_types_are_sources(self, mock_zotero_db: Path):
        """Anything that is not a note/attachment/annotation is indexable."""
        conn = sqlite3.connect(mock_zotero_db)
        conn.execute("INSERT INTO itemTypes (itemTypeID, typeName) VALUES (20, 'document')")
        conn.commit()
        conn.close()

        _add_item(mock_zotero_db, 202, "A_DOCUMENT", 20)
        _add_item(mock_zotero_db, 203, "DOC_ATTACH", ATTACHMENT)
        _add_pdf(mock_zotero_db, 203, 202)

        assert "A_DOCUMENT" in _item_keys(mock_zotero_db)
