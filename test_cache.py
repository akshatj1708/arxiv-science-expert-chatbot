"""
Test script to verify the caching implementation.
"""
import os
import time
import json
import hashlib
from pathlib import Path
import pandas as pd
import streamlit as st

# Import our modules
from utils.cache_utils import memory_cache, disk_cache, clear_all_caches, get_cache_info
from utils.data_processing import ArxivDataset, load_arxiv_data
from utils.nlp_utils import get_paper_summary, get_concept_explanation
from utils.pdf_utils import PDFProcessor

def test_memory_cache():
    """Test the memory cache decorator."""
    print("\n=== Testing Memory Cache ===")
    
    @memory_cache
    def expensive_operation(x):
        print(f"  Computing expensive_operation({x})...")
        time.sleep(1)  # Simulate work
        return x * x
    
    # First call - should compute
    start = time.time()
    result1 = expensive_operation(5)
    duration1 = time.time() - start
    print(f"  First call (should compute): {result1} in {duration1:.2f}s")
    
    # Second call with same args - should use cache
    start = time.time()
    result2 = expensive_operation(5)
    duration2 = time.time() - start
    print(f"  Second call (should use cache): {result2} in {duration2:.2f}s")
    
    # Different args - should compute again
    start = time.time()
    result3 = expensive_operation(10)
    duration3 = time.time() - start
    print(f"  Different args (should compute): {result3} in {duration3:.2f}s")
    
    assert result1 == 25
    assert result2 == 25
    assert result3 == 100
    assert duration2 < duration1  # Cached should be faster
    print("✅ Memory cache test passed!")

def test_disk_cache():
    """Test the disk cache decorator."""
    print("\n=== Testing Disk Cache ===")
    
    @disk_cache(ttl=5)  # 5 second TTL
    def expensive_disk_operation(x):
        print(f"  Computing expensive_disk_operation({x})...")
        time.sleep(1)  # Simulate work
        return {"result": x * x, "timestamp": time.time()}
    
    # First call - should compute
    start = time.time()
    result1 = expensive_disk_operation(5)
    duration1 = time.time() - start
    print(f"  First call (should compute): {result1} in {duration1:.2f}s")
    
    # Second call with same args - should use cache
    start = time.time()
    result2 = expensive_disk_operation(5)
    duration2 = time.time() - start
    print(f"  Second call (should use cache): {result2} in {duration2:.2f}s")
    
    # Different args - should compute again
    start = time.time()
    result3 = expensive_disk_operation(10)
    duration3 = time.time() - start
    print(f"  Different args (should compute): {result3} in {duration3:.2f}s")
    
    # Test TTL - wait for cache to expire
    print("  Waiting for cache to expire...")
    time.sleep(6)
    start = time.time()
    result4 = expensive_disk_operation(5)
    duration4 = time.time() - start
    print(f"  After TTL (should compute): {result4} in {duration4:.2f}s")
    
    assert result1["result"] == 25
    assert result2["result"] == 25
    assert result3["result"] == 100
    assert result4["result"] == 25
    assert duration2 < duration1  # Cached should be faster
    assert duration4 > duration2  # After TTL should be slower again
    print("✅ Disk cache test passed!")

def test_arxiv_data_caching():
    """Test caching with actual arXiv data functions."""
    print("\n=== Testing ArXiv Data Caching ===")
    
    # Test with a small category to avoid too much download
    category = "cs.AI"
    max_results = 2
    
    # First call - should download
    print(f"  Downloading {max_results} papers from {category}...")
    start = time.time()
    papers1 = load_arxiv_data(category, max_results=max_results, force_download=True)
    duration1 = time.time() - start
    print(f"  First download: {len(papers1)} papers in {duration1:.2f}s")
    
    # Second call - should use cache
    start = time.time()
    papers2 = load_arxiv_data(category, max_results=max_results, force_download=False)
    duration2 = time.time() - start
    print(f"  Second call (cached): {len(papers2)} papers in {duration2:.2f}s")
    
    # Force refresh
    start = time.time()
    papers3 = load_arxiv_data(category, max_results=max_results, force_download=True)
    duration3 = time.time() - start
    print(f"  Force refresh: {len(papers3)} papers in {duration3:.2f}s")
    
    assert len(papers1) > 0
    assert len(papers1) == len(papers2) == len(papers3)
    assert duration2 < duration1  # Cached should be faster
    print("✅ ArXiv data caching test passed!")

def test_nlp_caching():
    """Test caching of NLP operations."""
    print("\n=== Testing NLP Caching ===")
    
    test_text = """
    Machine learning is a method of data analysis that automates analytical model building. 
    It is a branch of artificial intelligence based on the idea that systems can learn 
    from data, identify patterns and make decisions with minimal human intervention.
    """
    
    # First call - should compute
    print("  Getting summary...")
    start = time.time()
    summary1 = get_paper_summary({"summary": test_text, "title": "Test Paper"})
    duration1 = time.time() - start
    print(f"  First summary in {duration1:.2f}s: {summary1[:50]}...")
    
    # Second call - should use cache
    start = time.time()
    summary2 = get_paper_summary({"summary": test_text, "title": "Test Paper"})
    duration2 = time.time() - start
    print(f"  Second summary (cached) in {duration2:.2f}s: {summary2[:50]}...")
    
    # Test concept explanation
    print("\n  Getting concept explanation...")
    start = time.time()
    explanation1 = get_concept_explanation("machine learning", test_text)
    duration1 = time.time() - start
    print(f"  First explanation in {duration1:.2f}s: {explanation1[:50]}...")
    
    # Second call - should use cache
    start = time.time()
    explanation2 = get_concept_explanation("machine learning", test_text)
    duration2 = time.time() - start
    print(f"  Second explanation (cached) in {duration2:.2f}s: {explanation2[:50]}...")
    
    assert summary1 == summary2
    assert explanation1 == explanation2
    print("✅ NLP caching test passed!")

def test_cache_management():
    """Test cache management functions."""
    print("\n=== Testing Cache Management ===")
    
    # Get initial cache info
    initial_info = get_cache_info()
    print(f"  Initial cache info: {json.dumps(initial_info, indent=2)}")
    
    # Clear all caches
    print("  Clearing all caches...")
    clear_all_caches()
    
    # Get cache info after clearing
    cleared_info = get_cache_info()
    print(f"  After clearing: {json.dumps(cleared_info, indent=2)}")
    
    # The memory cache should be empty, disk cache might have some system files
    assert cleared_info['memory_cache']['total_items'] == 0
    print("✅ Cache management test passed!")

def main():
    """Run all cache tests."""
    print("🔍 Starting Cache Tests 🔍")
    
    try:
        test_memory_cache()
        test_disk_cache()
        test_arxiv_data_caching()
        test_nlp_caching()
        test_cache_management()
        
        print("\n🎉 All cache tests passed successfully! 🎉")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
    
    # If running in Streamlit, show a nice UI
    try:
        import streamlit as st
        
        st.title("🧪 Cache Testing")
        st.write("Running cache tests...")
        
        # Capture the test output
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Run tests
        success = main()
        
        # Get the output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Display results
        if success:
            st.success("✅ All cache tests passed!")
        else:
            st.error("❌ Some tests failed. See output below.")
        
        st.code(output, language="text")
        
    except ImportError:
        pass  # Not running in Streamlit, just use console output
