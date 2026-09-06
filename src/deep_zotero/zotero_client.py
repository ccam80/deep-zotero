"""Zotero SQLite database client."""
import sqlite3
from pathlib import Path
from .models import ZoteroItem

# itemTypeID values come from Zotero's globalSchema and change between
# versions, so item types are matched by name.
NON_BIBLIOGRAPHIC_TYPE_IDS = """
    SELECT itemTypeID FROM itemTypes
    WHERE typeName IN ('note', 'attachment', 'annotation')
"""


class ZoteroClient:
    """
    Read-only access to Zotero's SQLite database.

    Key schema notes:
    - EAV pattern: itemData + itemDataValues + fields tables
    - Attachments: linkMode 0,1,4 = storage/{key}/, linkMode 2 = linked file
    - Citation keys come from Zotero's native citationKey field
    """

    # Combined query: items with PDFs and all metadata
    ITEMS_WITH_PDFS_SQL = f"""
    WITH
        base_items AS (
            SELECT items.itemID, items."key" AS itemKey, items.itemTypeID
            FROM items
            WHERE items.itemTypeID NOT IN ({NON_BIBLIOGRAPHIC_TYPE_IDS})
              AND items.itemID NOT IN (SELECT itemID FROM deletedItems)
        ),
        titles AS (
            SELECT itemData.itemID, itemDataValues.value AS title
            FROM itemData
            JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
            JOIN fields ON itemData.fieldID = fields.fieldID
            WHERE fields.fieldName = 'title'
        ),
        years AS (
            SELECT itemData.itemID, CAST(substr(itemDataValues.value, 1, 4) AS INTEGER) AS year
            FROM itemData
            JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
            JOIN fields ON itemData.fieldID = fields.fieldID
            WHERE fields.fieldName = 'date'
        ),
        authors AS (
            SELECT
                items.itemID,
                CASE
                    WHEN COUNT(*) = 1 THEN
                        MAX(creators.lastName) ||
                        CASE WHEN MAX(creators.firstName) IS NOT NULL AND MAX(creators.firstName) != ''
                             THEN ', ' || substr(MAX(creators.firstName), 1, 1) || '.'
                             ELSE '' END
                    ELSE
                        MAX(CASE WHEN itemCreators.orderIndex = 0 THEN creators.lastName END) || ' et al.'
                END AS authors
            FROM items
            JOIN itemCreators ON items.itemID = itemCreators.itemID
            JOIN creators ON itemCreators.creatorID = creators.creatorID
            GROUP BY items.itemID
        ),
        publications AS (
            SELECT itemData.itemID, itemDataValues.value AS publication
            FROM itemData
            JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
            JOIN fields ON itemData.fieldID = fields.fieldID
            WHERE fields.fieldName = 'publicationTitle'
        ),
        dois AS (
            SELECT itemData.itemID, itemDataValues.value AS doi
            FROM itemData
            JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
            JOIN fields ON itemData.fieldID = fields.fieldID
            WHERE fields.fieldName = 'DOI'
        ),
        citation_keys AS (
            SELECT itemData.itemID, itemDataValues.value AS citationKey
            FROM itemData
            JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
            JOIN fields ON itemData.fieldID = fields.fieldID
            WHERE fields.fieldName = 'citationKey'
        ),
        item_tags AS (
            SELECT items.itemID, GROUP_CONCAT(tags.name, '; ') AS tags
            FROM items
            JOIN itemTags ON items.itemID = itemTags.itemID
            JOIN tags ON itemTags.tagID = tags.tagID
            GROUP BY items.itemID
        ),
        item_collections AS (
            SELECT items.itemID, GROUP_CONCAT(c.collectionName, '; ') AS collection_names
            FROM items
            JOIN collectionItems ci ON items.itemID = ci.itemID
            JOIN collections c ON ci.collectionID = c.collectionID
            GROUP BY items.itemID
        ),
        pdfs AS (
            SELECT
                COALESCE(ia.parentItemID, ia.itemID) AS parentItemID,
                ia.itemID AS attachmentID,
                items."key" AS attachmentKey,
                ia.linkMode,
                ia.path
            FROM itemAttachments ia
            JOIN items ON ia.itemID = items.itemID
            WHERE ia.contentType = 'application/pdf'
              AND ia.linkMode IN (0, 1, 2)
        )
    SELECT
        base_items.itemKey,
        COALESCE(titles.title, '[No Title]') AS title,
        COALESCE(authors.authors, '[No Author]') AS authors,
        years.year,
        COALESCE(publications.publication, '') AS publication,
        COALESCE(dois.doi, '') AS doi,
        COALESCE(citation_keys.citationKey, '') AS citationKey,
        COALESCE(item_tags.tags, '') AS tags,
        COALESCE(item_collections.collection_names, '') AS collections,
        pdfs.attachmentKey,
        pdfs.linkMode,
        pdfs.path
    FROM base_items
    LEFT JOIN titles ON base_items.itemID = titles.itemID
    LEFT JOIN years ON base_items.itemID = years.itemID
    LEFT JOIN authors ON base_items.itemID = authors.itemID
    LEFT JOIN publications ON base_items.itemID = publications.itemID
    LEFT JOIN dois ON base_items.itemID = dois.itemID
    LEFT JOIN citation_keys ON base_items.itemID = citation_keys.itemID
    LEFT JOIN item_tags ON base_items.itemID = item_tags.itemID
    LEFT JOIN item_collections ON base_items.itemID = item_collections.itemID
    JOIN pdfs ON base_items.itemID = pdfs.parentItemID
    ORDER BY base_items.itemID, pdfs.attachmentID;
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "zotero.sqlite"
        if not self.db_path.exists():
            raise FileNotFoundError(f"Zotero database not found: {self.db_path}")

    def _resolve_pdf_path(self, path_field: str | None, link_mode: int, attachment_key: str) -> Path | None:
        """
        Resolve attachment path based on linkMode.

        Link modes (from Zotero source):
        - 0: IMPORTED_FILE - storage/{attachmentKey}/{filename}
        - 1: IMPORTED_URL  - storage/{attachmentKey}/{filename}
        - 2: LINKED_FILE   - relative to linked attachment base dir (skip for now)
        - 3: LINKED_URL    - no local file
        - 4: EMBEDDED_IMAGE - storage/{attachmentKey}/{filename}
        """
        if path_field is None:
            return None

        if link_mode == 2:
            # Linked file - would need base dir from Zotero prefs
            # Skip for now, or make configurable
            return None

        if path_field.startswith("storage:"):
            filename = path_field[len("storage:"):]
            full_path = self.data_dir / "storage" / attachment_key / filename
            return full_path if full_path.exists() else None

        return None

    def get_all_items_with_pdfs(self) -> list[ZoteroItem]:
        """One item per Zotero item, using its oldest PDF attachment that exists on disk."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(self.ITEMS_WITH_PDFS_SQL)
            rows = cursor.fetchall()
        finally:
            conn.close()

        items: dict[str, ZoteroItem] = {}
        for row in rows:
            pdf_path = self._resolve_pdf_path(
                row["path"],
                row["linkMode"],
                row["attachmentKey"]
            )
            existing = items.get(row["itemKey"])
            if existing is not None and (existing.pdf_path is not None or pdf_path is None):
                continue
            items[row["itemKey"]] = ZoteroItem(
                item_key=row["itemKey"],
                title=row["title"],
                authors=row["authors"],
                year=row["year"],
                pdf_path=pdf_path,
                citation_key=row["citationKey"],
                publication=row["publication"],
                doi=row["doi"],
                tags=row["tags"],
                collections=row["collections"],
            )

        return list(items.values())

    def get_item(self, item_key: str) -> ZoteroItem | None:
        """Get a specific item by key."""
        # For now, just filter from all items
        # Could optimize with a WHERE clause if needed
        all_items = self.get_all_items_with_pdfs()
        for item in all_items:
            if item.item_key == item_key:
                return item
        return None

