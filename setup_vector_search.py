#!/usr/bin/env python3
"""
Setup script for migrating to vector-based search in the CS2 Skin Economy API.
This script will install dependencies, initialize embeddings, and test the system.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🔄 {description or command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing vector search dependencies...")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("⚠️  Warning: Not in a virtual environment. Consider using one.")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing all requirements"):
        return False
    
    # Test critical imports
    print("\n🧪 Testing critical imports...")
    critical_imports = [
        "sentence_transformers",
        "faiss",
        "numpy",
        "sklearn"
    ]
    
    for module in critical_imports:
        try:
            __import__(module.replace('-', '_'))
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    print("\n📂 Checking data files...")
    
    data_dir = Path("data")
    skinport_file = data_dir / "skinport_data.json"
    docs_dir = data_dir / "documents"
    
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    if not skinport_file.exists():
        print("❌ skinport_data.json not found")
        return False
    
    # Check file size
    file_size = skinport_file.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ skinport_data.json found ({file_size:.1f} MB)")
    
    if not docs_dir.exists():
        print("⚠️  Documents directory not found - creating it")
        docs_dir.mkdir(exist_ok=True)
    
    # Count documents
    doc_files = list(docs_dir.glob("*.txt"))
    print(f"📄 Found {len(doc_files)} document files")
    
    return True

def test_vector_search():
    """Test the vector search system"""
    print("\n🔍 Testing vector search system...")
    
    try:
        # Test vector search engine
        from vector_search_engine import get_vector_skin_search_engine
        search_engine = get_vector_skin_search_engine()
        print(f"✅ Vector search engine initialized with {len(search_engine.item_names)} items")
        
        # Test a sample search
        test_query = "AK-47 skins"
        results = search_engine.search(test_query, limit=3)
        print(f"✅ Sample search for '{test_query}' returned {len(results)} results")
        
        if results:
            print("📋 Sample results:")
            for i, result in enumerate(results[:2], 1):
                name = result.get('item_name', 'Unknown')
                score = result.get('relevance_score', 0)
                print(f"  {i}. {name} (relevance: {score:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing vector search: {e}")
        return False

def test_enhanced_documents():
    """Test enhanced document processing"""
    print("\n📚 Testing enhanced document processing...")
    
    try:
        from enhanced_document_tools import create_enhanced_document_tool
        doc_tool = create_enhanced_document_tool()
        print("✅ Enhanced document processor initialized")
        
        # Test a sample query
        test_query = "CS2 trading strategies"
        result = doc_tool.func(test_query)
        print(f"✅ Sample document query completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced documents: {e}")
        return False

def create_cache_directories():
    """Create necessary cache directories"""
    print("\n📁 Creating cache directories...")
    
    cache_dirs = [
        "data/doc_cache",
        "data/embeddings"
    ]
    
    for cache_dir in cache_dirs:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {cache_dir}")
    
    return True

def generate_embeddings():
    """Pre-generate embeddings for faster startup"""
    print("\n⚡ Pre-generating embeddings (this may take a few minutes)...")
    
    try:
        # Initialize vector search to create embeddings
        from vector_search_engine import get_vector_skin_search_engine
        search_engine = get_vector_skin_search_engine()
        print("✅ Skin embeddings generated and cached")
        
        # Initialize document embeddings
        from enhanced_document_tools import create_enhanced_document_tool
        doc_tool = create_enhanced_document_tool()
        print("✅ Document embeddings generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return False

def run_comparison_test():
    """Run a comparison between old and new search methods"""
    print("\n🆚 Running search comparison test...")
    
    test_queries = [
        "cheap AK-47 skins",
        "best knife under $100",
        "fade pattern weapons",
        "StatTrak rifles"
    ]
    
    try:
        # Test vector search
        from vector_search_engine import get_vector_skin_search_engine
        vector_engine = get_vector_skin_search_engine()
        
        # Test fallback search
        from search_utils_simplified import get_skin_search_engine
        basic_engine = get_skin_search_engine()
        
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            
            # Vector results
            vector_results = vector_engine.search(query, limit=3)
            print(f"  Vector search: {len(vector_results)} results")
            
            # Basic results
            basic_results = basic_engine.search(query, limit=3)
            print(f"  Basic search: {len(basic_results)} results")
            
        print("✅ Comparison test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error in comparison test: {e}")
        return False

def create_startup_script():
    """Create a startup script for the vector-enhanced system"""
    startup_script = """#!/bin/bash
echo "Starting CS2 Skin Economy API with Vector Search..."

# Check if we're in the right directory
if [ ! -f "main_vector.py" ]; then
    echo "Error: main_vector.py not found. Please run from the 2m-backend directory."
    exit 1
fi

# Start the server
echo "🚀 Starting vector-enhanced API server..."
python main_vector.py
"""
    
    with open("start_vector_backend.sh", "w") as f:
        f.write(startup_script)
    
    # Make it executable
    os.chmod("start_vector_backend.sh", 0o755)
    print("✅ Created start_vector_backend.sh script")

def main():
    """Main setup function"""
    print("🎯 CS2 Skin Economy API - Vector Search Setup")
    print("=" * 60)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing dependencies", install_dependencies),
        ("Checking data files", check_data_files),
        ("Creating cache directories", create_cache_directories),
        ("Generating embeddings", generate_embeddings),
        ("Testing vector search", test_vector_search),
        ("Testing enhanced documents", test_enhanced_documents),
        ("Running comparison test", run_comparison_test),
        ("Creating startup script", create_startup_script)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        try:
            if not step_func():
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ Unexpected error in {step_name}: {e}")
            failed_steps.append(step_name)
    
    print("\n" + "=" * 60)
    print("🏁 SETUP COMPLETE")
    print("=" * 60)
    
    if failed_steps:
        print("⚠️  Some steps failed:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nYou can still try running the system, but some features may not work.")
    else:
        print("✅ All steps completed successfully!")
    
    print("\n📋 Next Steps:")
    print("1. Run 'python main_vector.py' to start with vector search")
    print("2. Or run './start_vector_backend.sh' to use the startup script")
    print("3. Visit http://localhost:8000 to check API status")
    print("4. Your frontend should automatically benefit from improved search!")
    
    print("\n🔍 Search Improvements:")
    print("- Semantic understanding of queries")
    print("- Better intent recognition")
    print("- Context-aware results")
    print("- Relevance scoring")
    print("- Enhanced document processing")

if __name__ == "__main__":
    main() 