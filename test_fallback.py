#!/usr/bin/env python3
"""
Test script for the CS2 skin search engine fallback mechanism.
This script demonstrates the fallback search capabilities for different query types.
"""

import os
import json

# Try importing from fallback directly to test it
from search_utils_fallback import get_skin_search_engine

def test_fallback_search():
    """Test the fallback search engine with various queries"""
    # Initialize the search engine
    print("Initializing fallback search engine...")
    search_engine = get_skin_search_engine()
    print(f"Search engine initialized with {len(search_engine.item_names)} items")
    
    # Test queries
    test_queries = [
        "Hedge Maze gloves",
        "Sport Gloves Hedge Maze",
        "hedge mase glove",  # Intentional typo
        "AWP Dragon Lore",
        "Karambit fade",
        "butterfly knife",
        "m4a4 howl",
        "asiimov",
    ]
    
    print("\n" + "="*60)
    print(f"{'QUERY':<30} | {'TOP MATCHES'}")
    print("="*60)
    
    for query in test_queries:
        # First show the expanded query
        expanded = search_engine._expand_query(query)
        print(f"{query:<30} | Expanded: {expanded}")
        
        # Then show search results
        results = search_engine.hybrid_search(query, top_k=3)
        formatted_results = []
        
        for result in results:
            score_info = f"(score: {result['total_score']:.2f})"
            formatted_results.append(f"{result['item_name']} {score_info}")
        
        result_str = ", ".join(formatted_results)
        print(f"{'  Results:':<30} | {result_str}")
        print("-"*60)
    
    print("="*60)

if __name__ == "__main__":
    test_fallback_search() 