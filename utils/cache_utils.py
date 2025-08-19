"""
Caching utilities for the application.
Uses functools.lru_cache for in-memory caching and diskcache for persistent caching.
"""
import functools
import hashlib
import inspect
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, cast

from diskcache import Cache

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])

# Global cache directory
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_key(func: Callable, *args, **kwargs) -> str:
    """Generate a unique cache key for function arguments."""
    # Create a unique key based on function name and arguments
    args_repr = [repr(arg) for arg in args]
    kwargs_repr = [f"{k}={v!r}" for k, v in sorted(kwargs.items())]
    key_parts = [func.__module__, func.__name__] + args_repr + kwargs_repr
    key_string = ":".join(key_parts)
    
    # Create a hash of the key string to ensure it's a valid filename
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

def memory_cache(maxsize: int = 128) -> Callable[[F], F]:
    """
    Decorator for in-memory function result caching using LRU strategy.
    
    Args:
        maxsize: Maximum number of entries to keep in the cache.
    """
    def decorator(func: F) -> F:
        @functools.lru_cache(maxsize=maxsize)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator

def disk_cache(
    namespace: Optional[str] = None,
    ttl: int = 3600,
    max_size: int = 2**30,  # 1GB
    **cache_kwargs
) -> Callable[[F], F]:
    """
    Decorator for disk-based function result caching.
    
    Args:
        namespace: Namespace for the cache. If None, uses function's module and name.
        ttl: Time to live in seconds for cache entries.
        max_size: Maximum size of the cache in bytes.
        **cache_kwargs: Additional arguments to pass to diskcache.Cache.
    """
    def decorator(func: F) -> F:
        nonlocal namespace
        
        if namespace is None:
            namespace = f"{func.__module__}.{func.__qualname__}"
        
        # Create a cache directory for this namespace
        cache_dir = CACHE_DIR / namespace
        cache = Cache(
            directory=str(cache_dir.absolute()),
            size_limit=max_size,
            **cache_kwargs
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Skip caching for methods that are not safe to cache
            if any(arg_name in kwargs for arg_name in ['no_cache', 'skip_cache']):
                if kwargs.pop('no_cache', False) or kwargs.pop('skip_cache', False):
                    return func(*args, **kwargs)
            
            # Generate cache key
            key = get_cache_key(func, *args, **kwargs)
            
            # Try to get from cache
            try:
                result = cache.get(key)
                if result is not None:
                    return result
            except Exception as e:
                # If cache read fails, continue with function execution
                print(f"Cache read error: {e}")
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Store in cache
            try:
                cache.set(key, result, expire=ttl)
            except Exception as e:
                print(f"Cache write error: {e}")
            
            return result
        
        # Add cache clearing method
        def clear_cache():
            """Clear the cache for this function."""
            cache.clear()
        
        wrapper.clear_cache = clear_cache  # type: ignore
        wrapper.cache = cache  # type: ignore
        
        return cast(F, wrapper)
    
    return decorator

def clear_all_caches():
    """Clear all disk caches."""
    for item in CACHE_DIR.glob("*"):
        if item.is_dir():
            cache = Cache(directory=str(item.absolute()))
            cache.clear()
            cache.close()

def get_cache_info() -> dict:
    """Get information about all caches."""
    cache_info = {}
    for item in CACHE_DIR.glob("*"):
        if item.is_dir():
            cache = Cache(directory=str(item.absolute()))
            cache_info[item.name] = {
                'size': cache.volume(),
                'count': len(cache),
                'path': str(item.absolute())
            }
            cache.close()
    return cache_info
