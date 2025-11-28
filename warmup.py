#!/usr/bin/env python3
"""Warmup script to preload models and indexes on container startup."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, '/app')

print("🔥 Starting warmup...")

try:
    # Import and trigger model loading
    print("  ├─ Loading PDF query tools...")
    from tools import pdf_query_tools
    
    # Trigger embeddings and FAISS loading
    print("  ├─ Warming up Constitution index...")
    pdf_query_tools.indian_constitution_pdf_query.invoke("preamble")
    
    print("  ├─ Warming up BNS Laws index...")
    pdf_query_tools.indian_laws_pdf_query.invoke("section 1")
    
    # Trigger agent executor creation
    print("  ├─ Initializing agent...")
    from agent import _get_agent_executor
    _get_agent_executor()
    
    print("✅ Warmup complete! System ready for queries.")
    
except Exception as e:
    print(f"⚠️  Warmup warning: {e}")
    print("System will warm up on first query instead.")
