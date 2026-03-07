"""CLI entry point for indexing Zotero libraries."""
import argparse
import logging
import sys

from .config import Config
from .indexer import Indexer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zotero-index",
        description="Index Zotero PDFs into the chunk-RAG vector store.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-index all matching items (delete and rebuild)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of items to index",
    )
    parser.add_argument(
        "--item-key", type=str, default=None,
        help="Index only this specific Zotero item key",
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Regex pattern to filter items by title (case-insensitive)",
    )
    parser.add_argument(
        "--no-vision", action="store_true",
        help="Disable vision-based table extraction even if configured",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config JSON file (default: ~/.config/zotero-chunk-rag/config.json)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    config = Config.load(args.config)
    errors = config.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}", file=sys.stderr)
        return 1

    if args.no_vision:
        config.vision_enabled = False

    indexer = Indexer(config)
    result = indexer.index_all(
        force_reindex=args.force,
        limit=args.limit,
        item_key=args.item_key,
        title_pattern=args.title,
    )

    # Print summary
    print(f"\nIndexing complete:")
    print(f"  Indexed:         {result['indexed']}")
    print(f"  Already indexed: {result['already_indexed']}")
    print(f"  Skipped (empty): {result['skipped']}")
    print(f"  Failed:          {result['failed']}")
    print(f"  Empty:           {result['empty']}")

    if result.get("quality_distribution"):
        dist = result["quality_distribution"]
        print(f"  Quality: A={dist.get('A',0)} B={dist.get('B',0)} "
              f"C={dist.get('C',0)} D={dist.get('D',0)} F={dist.get('F',0)}")

    if result.get("extraction_stats"):
        stats = result["extraction_stats"]
        print(f"  Pages: {stats.get('total_pages',0)} total, "
              f"{stats.get('text_pages',0)} text, "
              f"{stats.get('ocr_pages',0)} OCR, "
              f"{stats.get('empty_pages',0)} empty")

    # Print failures
    failures = [r for r in result["results"] if r.status == "failed"]
    if failures:
        print(f"\nFailures:")
        for f in failures:
            print(f"  {f.item_key}: {f.reason}")

    return 1 if result["failed"] > 0 and result["indexed"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
