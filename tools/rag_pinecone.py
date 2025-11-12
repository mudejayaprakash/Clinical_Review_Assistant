"""
PolicyMind RAG functions adapted for Pinecone with HYBRID SEARCH
"""
from pinecone import Pinecone
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from agents.config import PINECONE_API_KEY,PINECONE_INDEX_NAME,PINECONE_NAMESPACE,EMBEDDING_MODEL
import re

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize embeddings
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize CrossEncoder for re-ranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def expand_query(query: str, llm_client=None, llm_model: str = "gpt-4o-mini") -> list:
    """
    Expand query using LLM to generate medical synonyms and variations
    
    Args:
        query: Original search query
        llm_client: OpenAI client (optional)
        llm_model: Model to use for expansion (default: gpt-4o-mini for cost)
        
    Returns:
        List of query variations
    """

    variations = [query]

    # If no LLM client, return original query only
    if not llm_client:
        print(f"[Query Expansion] No LLM client provided, using original query only")
        return variations
    try:
        system_msg = (
            "You are a medical terminology assistant. "
            "Generate 3-5 medical synonyms, abbreviations, or clinically equivalent terms for the given query. "
            "Focus on terms that would appear in medical policies and insurance documentation.\n\n"
            "Return ONLY a comma-separated list of terms, no explanations or numbering.\n\n"
            "Examples:\n"
            "Input: 'nasal obstruction'\n"
            "Output: blocked nose, nasal blockage, nasal congestion, stuffy nose\n\n"
            "Input: 'deviated septum'\n"
            "Output: septal deviation, nasal septum deviation, deviated nasal septum"
        )
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Medical query: {query}"}
            ],
            temperature=0,  # Deterministic output
            max_tokens=100
        )

        text = response.choices[0].message.content.strip()

        # Parse comma-separated variations
        llm_variations = [v.strip() for v in text.split(",") if v.strip()]
        variations.extend(llm_variations)

        print(f"[Query Expansion] Original: '{query}'")
        print(f"[Query Expansion] LLM generated: {llm_variations}")
        
    except Exception as e:
        print(f"[Query Expansion] LLM expansion failed: {e}")
        print(f"[Query Expansion] Falling back to original query only")

    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        vl = v.lower()
        if vl not in seen:
            seen.add(vl)
            unique_variations.append(v)
    
    return unique_variations[:5]  # Limit to top 5 variations


def calculate_keyword_overlap(query: str, text: str) -> float:
    """
    Calculate keyword overlap score between query and text (BM25-like)
    
    Args:
        query: Search query
        text: Document text
        
    Returns:
        Overlap score (0-1)
    """
    # Tokenize and normalize
    query_terms = set(re.findall(r'\b\w+\b', query.lower()))
    text_terms = set(re.findall(r'\b\w+\b', text.lower()))
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    query_terms -= stop_words
    text_terms -= stop_words
    
    if not query_terms:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = query_terms & text_terms
    overlap_score = len(intersection) / len(query_terms)
    
    return overlap_score


def retrieve_top_contexts_pinecone(query: str, top_k: int = 50, cross_encoder_top_k: int = 20, use_hybrid: bool = True, 
                                   llm_client=None, llm_model: str = "gpt-4o-mini"):
    """
    HYBRID SEARCH: Retrieve and re-rank contexts from Pinecone
    Uses LLM-based query expansion + keyword matching + cross-encoder reranking
    
    Args:
        query: Search query
        top_k: Initial retrieval count per query variation
        cross_encoder_top_k: Number to keep after re-ranking
        use_hybrid: If True, use query expansion and keyword matching
        llm_client: OpenAI client for query expansion (optional)
        llm_model: Model to use for expansion (default: gpt-4o-mini)
        
    Returns:
        List of ranked chunks with metadata
    """
    all_matches = []
    
    if use_hybrid:
        # Step 1: Expand query with LLM-generated variations
        query_variations = expand_query(query, llm_client=llm_client, llm_model=llm_model)
        print(f"[Hybrid Search] Query variations: {query_variations}")
    else:
        query_variations = [query]

    
    # Step 2: Search with each query variation
    for q in query_variations:
        query_embedding = embedder.embed_query(q)
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k // len(query_variations),  # Split quota across variations
            include_metadata=True,
            namespace=PINECONE_NAMESPACE
        )
        
        matches = results.get('matches', [])
        all_matches.extend(matches)
    
    if not all_matches:
        print(f"[Hybrid Search] No matches found for query: {query}")
        return []
    
    # Remove duplicates based on ID
    unique_matches = {}
    for match in all_matches:
        if match.id not in unique_matches:
            unique_matches[match.id] = match
    
    all_matches = list(unique_matches.values())
    print(f"[Hybrid Search] Retrieved {len(all_matches)} unique chunks")
    
    # Step 3: Re-rank with CrossEncoder + keyword overlap
    pairs = [[query, match.metadata.get('text', '')] for match in all_matches]
    cross_scores = cross_encoder.predict(pairs)
    
    # Calculate keyword overlap scores
    if use_hybrid:
        keyword_scores = [calculate_keyword_overlap(query, match.metadata.get('text', '')) 
                         for match in all_matches]
    else:
        keyword_scores = [0.0] * len(all_matches)
    
    # Combine scores: 70% cross-encoder + 30% keyword overlap
    for i, match in enumerate(all_matches):
        combined_score = 0.7 * float(cross_scores[i]) + 0.3 * keyword_scores[i]
        match.metadata['similarity_score'] = combined_score
        match.metadata['cross_score'] = float(cross_scores[i])
        match.metadata['keyword_score'] = keyword_scores[i]
    
    # Sort by combined score
    ranked_matches = sorted(all_matches, key=lambda x: x.metadata.get('similarity_score', 0), reverse=True)
    
    # Take top_k after re-ranking
    top_matches = ranked_matches[:cross_encoder_top_k]
    
    print(f"[Hybrid Search] Top match score: {top_matches[0].metadata.get('similarity_score', 0):.3f} (cross: {top_matches[0].metadata.get('cross_score', 0):.3f}, keyword: {top_matches[0].metadata.get('keyword_score', 0):.3f})")
    
    # Format to match original rag.py output format
    formatted_chunks = []
    for match in top_matches:
        formatted_chunks.append({
            "page_content": match.metadata.get('text', ''),
            "metadata": {
                "policy_id": match.metadata.get('policy_id', ''),
                "source": match.metadata.get('source', ''),
                "page": match.metadata.get('page', 1),
                "section": match.metadata.get('section', '')
            },
            "similarity_score": match.metadata.get('similarity_score', 0.0)
        })
    
    return formatted_chunks


def _build_citation_map(retrieved_chunks):
    """
    Build citation map from chunks - ORIGINAL FUNCTION from rag.py
    
    Args:
        retrieved_chunks: List of Document objects
        
    Returns:
        Tuple of (context_text, citation_map)
    """
    citation_map = {}
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks, start=1):
        # Get metadata
        metadata = chunk.metadata if hasattr(chunk, 'metadata') else chunk.get('metadata', {})
        page = metadata.get("page", "N/A")
        source = metadata.get("source", "Unknown")
        
        # Get content
        text = chunk.page_content if hasattr(chunk, 'page_content') else chunk.get('page_content', '')
        
        # Build citation
        citation_map[f"[{i}]"] = {
            "page": page,
            "source": source,
            "text": text[:500]  # Store preview
        }
        
        # Add to context
        context_parts.append(f"[{i}] (Source: {source}, Page: {page})\n{text}\n")
    
    context_text = "\n".join(context_parts)
    return context_text, citation_map


def summarize_policy_chunks(retrieved_chunks, llm_client, llm_model: str, top_k: int = 5):
    """
    Summarize policy chunks with inline citations - ORIGINAL PROVEN PROMPT from rag.py
    
    Args:
        retrieved_chunks: List of retrieved Document objects
        llm_client: OpenAI client
        llm_model: Model name
        top_k: Number of top chunks to use
        
    Returns:
        Dict with "summary" and "references"
    """
    # Take top K chunks
    top_chunks = retrieved_chunks[:top_k]
    
    # Build citation map using ORIGINAL function
    context_text, citation_map = _build_citation_map(top_chunks)
    
    # ORIGINAL PROVEN SYSTEM PROMPT from rag.py
    system_msg = (
        "You are a healthcare policy expert. Summarize the provided policy excerpts clearly and concisely.\n\n"
        "**CRITICAL CITATION RULES**:\n"
        "- Every fact MUST have an inline citation in brackets: [1], [2], [3], etc.\n"
        "- Citations refer to the numbered chunks in the context below\n"
        "- Format: 'Requirement or fact [citation_number]'\n"
        "- Multiple related facts can share citations or use ranges: [1-3]\n"
        "- If information is not in the context, do not mention it\n"
        "- Structure summary with clear sections using markdown headers\n\n"
        "**EXAMPLE FORMAT**:\n"
        "### Coverage Requirements\n"
        "- Prior authorization required [1]\n"
        "- Documentation of medical necessity [2]\n"
        "- Failed conservative therapy for 6 months [3]\n\n"
        "**Exclusions**\n"
        "- Experimental treatments not covered [4]\n"
        "- Cosmetic procedures excluded [5]\n\n"
        "Remember: Every bullet point needs a [number] citation!"
    )
    
    user_msg = f"Policy Context:\n{context_text}\n\nSummarize the key criteria clearly with citations."
    
    try:
        # Use ORIGINAL few-shot example from rag.py
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_msg},
                # Few-shot example showing citation format
                {"role": "user", "content": "Summarize this policy: [1] Requires authorization. [2] Must document need. [3] Trial therapy for 6 months required."},
                {"role": "assistant", "content": "**Requirements**\n- Authorization required [1]\n- Documentation of medical need [2]\n- 6-month trial of conservative therapy [3]"},
                # Actual user request
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=500,
        )
        summary_text = response.choices[0].message.content.strip()
        
        return {
            "summary": summary_text,
            "references": citation_map
        }
    
    except Exception as e:
        print("Summarization failed:", e)
        import traceback
        traceback.print_exc()
        return {
            "summary": "Unable to summarize the policy at this time. Please try again later.",
            "references": {}
        }


def check_pinecone_status():
    """
    Diagnostic function to check Pinecone index status
    Call this if you're getting 0 results
    """
    try:
        stats = index.describe_index_stats()
        print(f"\n=== Pinecone Index Status ===")
        print(f"Index Name: {PINECONE_INDEX_NAME}")
        print(f"Namespace: {PINECONE_NAMESPACE}")
        print(f"Total Vectors: {stats.get('total_vector_count', 0)}")
        print(f"Dimension: {stats.get('dimension', 'N/A')}")
        
        if 'namespaces' in stats:
            print(f"\nNamespace breakdown:")
            for ns_name, ns_stats in stats['namespaces'].items():
                print(f"  - {ns_name}: {ns_stats.get('vector_count', 0)} vectors")
        
        if stats.get('total_vector_count', 0) == 0:
            print("\n⚠️ WARNING: Index is empty! Run data_ingestion.py to populate it.")
        elif PINECONE_NAMESPACE not in stats.get('namespaces', {}):
            print(f"\n⚠️ WARNING: Namespace '{PINECONE_NAMESPACE}' not found!")
            print(f"   Available namespaces: {list(stats.get('namespaces', {}).keys())}")
        else:
            print(f"\n✅ Index is ready with data in correct namespace")
        
        return stats
    except Exception as e:
        print(f"Error checking Pinecone status: {e}")
        import traceback
        traceback.print_exc()
        return None
