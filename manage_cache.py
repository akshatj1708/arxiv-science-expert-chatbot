"""
Cache management commands for the arXiv Research Assistant.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

from utils.cache_utils import clear_all_caches, get_cache_info

def clear_cache(args: argparse.Namespace) -> None:
    """Clear all caches or a specific cache."""
    if args.namespace:
        # Clear a specific namespace
        cache_dir = Path(".cache") / args.namespace
        if cache_dir.exists() and cache_dir.is_dir():
            from diskcache import Cache
            cache = Cache(str(cache_dir.absolute()))
            cache.clear()
            cache.close()
            print(f"Cleared cache for namespace: {args.namespace}")
        else:
            print(f"No cache found for namespace: {args.namespace}")
    else:
        # Clear all caches
        clear_all_caches()
        print("All caches cleared.")

def show_cache_info(args: argparse.Namespace) -> None:
    """Show cache information."""
    cache_info = get_cache_info()
    
    if not cache_info:
        print("No caches found.")
        return
    
    if args.json:
        print(json.dumps(cache_info, indent=2))
    else:
        print("\nCache Information:")
        print("-" * 80)
        for namespace, info in cache_info.items():
            print(f"Namespace: {namespace}")
            print(f"  Size: {info['size'] / (1024 * 1024):.2f} MB")
            print(f"  Items: {info['count']}")
            print(f"  Path: {info['path']}")
            print("-" * 80)

def main() -> None:
    """Main entry point for cache management commands."""
    parser = argparse.ArgumentParser(description="Manage arXiv Research Assistant caches")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Clear cache command
    clear_parser = subparsers.add_parser("clear", help="Clear caches")
    clear_parser.add_argument(
        "--namespace", 
        type=str, 
        help="Specific cache namespace to clear (default: all)"
    )
    clear_parser.set_defaults(func=clear_cache)
    
    # Show cache info command
    info_parser = subparsers.add_parser("info", help="Show cache information")
    info_parser.add_argument(
        "--json", 
        action="store_true", 
        help="Output in JSON format"
    )
    info_parser.set_defaults(func=show_cache_info)
    
    # Parse arguments and execute the appropriate function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
