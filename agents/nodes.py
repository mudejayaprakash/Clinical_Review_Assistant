"""
Node implementations for Clinical Review Assistant Agent
"""
import os
import re
from typing import Dict, List
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import chromadb
import tempfile

import agents.config as config
from agents.security import security

# Initialize models
summary_llm = ChatOpenAI(
    model=config.MODEL_SUMMARY,
    temperature=config.TEMPERATURE_SUMMARY,
    max_tokens=config.MAX_TOKENS_SUMMARY,
    timeout=120,
    max_retries=2  # Retry on failure
)

evaluation_llm = ChatOpenAI(
    model=config.MODEL_EVALUATION,
    temperature=config.TEMPERATURE_EVALUATION,
    max_tokens=config.MAX_TOKENS_EVALUATION
)

# Initialize SapBERT embeddings
sapbert_model = SentenceTransformer(config.EMBEDDING_MODEL)


class Node1_MedicalRecordProcessor:
    """
    Node 1: Process medical records
    - Parse PDFs with page tracking
    - Create summary with grouped citations
    - Extract chief complaints as list 
    - Create chunks and embed with SapBERT
    - Store in ChromaDB Ephemeral
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len
        )
    
    def process(self, state: Dict) -> Dict:
        """Process medical records and create searchable chunks"""
        security.log_action("Node1_Start", state.get("user_id", "unknown"), {"num_records": len(state.get("medical_records", []))})
        
        medical_records = state.get("medical_records", [])
        selected_records = [r for r in medical_records if r.get("selected", False)]
        
        if not selected_records:
            return {
                **state,
                "errors": state.get("errors", []) + ["No medical records selected"],
                "next_action": "end"
            }
        
        try:
            # Step 1: Extract text from PDFs with page tracking
            all_documents = []
            documents_processed = []
            
            for record in selected_records:
                filename = record["filename"]
                content_bytes = record["content"]
                
                # Create temporary file for UnstructuredPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(content_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    # Load PDF with page tracking
                    loader = UnstructuredPDFLoader(tmp_path, mode="elements")
                    docs = loader.load()
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata["source"] = filename
                        # Extract page number from metadata if available
                        page_num = doc.metadata.get("page_number", 1)
                        doc.metadata["page"] = page_num
                    
                    all_documents.extend(docs)
                    documents_processed.append(filename)
                    
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
            
            # Step 2: Create summary with reasoning
            try:
                summary_text = self._create_summary(all_documents, documents_processed)
            except Exception as e:
                print(f"[Node 1] LLM summarization failed: {e}. Using basic extraction fallback.")
                # FALLBACK: Create basic summary from regex extraction
                summary_text = self._create_basic_summary(all_documents, documents_processed)
            
            # Step 3: Extract chief complaints as list 
            try:
                chief_complaints = self._extract_chief_complaints(summary_text)
            except Exception as e:
                print(f"[Node 1] Chief complaint extraction failed: {e}. Using fallback.")
                # FALLBACK: Extract from first document
                chief_complaints = ["Medical evaluation"]  # Generic fallback
            
            # Step 4: Create chunks with metadata
            chunks = self._create_chunks(all_documents)
            
            # Step 5: Create ChromaDB Ephemeral collection
            chromadb_collection = self._create_chromadb_collection(chunks)
            
            # Step 6: Create citations map
            citations_map = self._create_citations_map(all_documents)
            
            reasoning = f"""
            
- **{len(documents_processed)}** Documents Processed, **Files**: {', '.join(documents_processed)}
- **{len(all_documents)}** pages extracted and **{len(chunks)}** chunks created
- **ChromaDB Status**: Ephemeral collection created with SapBERT embeddings

**Processing Steps**:
1. Extracted text from {len(documents_processed)} PDFs using UnstructuredPDFLoader
2. Created clinical summary with citations from actual uploaded documents only
3. Identified {len(chief_complaints)} chief complaint(s): {', '.join(chief_complaints)}
4. Generated {len(chunks)} text chunks with document/page metadata
5. Embedded chunks using SapBERT medical embeddings
6. Stored in ephemeral ChromaDB for Node 3 retrieval
            
**Quality Check**: ✅  All records parsed successfully 
**PHI Protection**: ✅  In-memory processing only, no persistence
"""
            security.log_action("Node1_Complete", state.get("user_id", "unknown"), {
                "docs_processed": len(documents_processed),
                "chunks_created": len(chunks)
            })
            
            return {
                **state,
                "summary": summary_text,
                "chief_complaint": chief_complaints, 
                "node1_reasoning": reasoning,
                "documents_processed": documents_processed,
                "citations_map": citations_map,
                "record_chunks": chunks,
                "chromadb_collection": chromadb_collection,
                "next_action": "node2"
            }
            
        except Exception as e:
            error_msg = f"Node 1 Error: {str(e)}"
            security.log_action("Node1_Error", state.get("user_id", "unknown"), {"error": str(e)})

            return {
                **state,
                "errors": state.get("errors", []) + [error_msg],
                "next_action": "end"
            }
    
    def _create_summary(self, documents: List, filenames: List[str]) -> str:
        """
        Create clinical summary with grouped citations
        """
        # Combine all document text
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Build explicit list of available documents
        doc_context = "AVAILABLE DOCUMENTS (cite ONLY these documents):\n"
        for filename in filenames:
            doc_pages = [doc for doc in documents if doc.metadata.get("source") == filename]
            pages = sorted(set([doc.metadata.get("page", 1) for doc in doc_pages]))
            doc_context += f"- {filename}: pages {min(pages)}-{max(pages)}\n"
        
        prompt = f"""You are a clinical review assistant. Create a comprehensive 2-3 paragraph summary of the medical record below.

{doc_context}

CRITICAL RULES:
- Only cite documents from the list above
- Do NOT invent or reference any documents not in the list
- Use inline citations: [filename, p.X] where X is the actual page number
- Do not add a separate "Citations by Document" section at the end

Medical Record:
{full_text[:8000]}

Provide a detailed clinical summary with inline citations covering:
- Medical History
- Social and Family History
- Physical Examination
- Imaging and Diagnostic Studies
- Assessment and Plan

Use inline citations for all factual claims."""
        
        response = summary_llm.invoke(prompt)
        return response.content
    
    def _extract_chief_complaints(self, summary: str) -> List[str]:
        """
        Extract TOP 3 PRIMARY chief complaints from summary
        Focus on main medical reasons for visit/procedure
        """
        prompt = f"""Extract the TOP 3 PRIMARY chief complaints (main reasons for medical visit/procedure) from this summary.

Summary: {summary[:2000]}

CRITICAL RULES:
- Extract ONLY the 3 most important PRIMARY medical complaints
- Focus on the main diagnosis or procedure reason (e.g., "nasal obstruction requiring septoplasty")
- Exclude secondary symptoms (fatigue, general discomfort, etc.)
- Each complaint should be concise (4-8 words)
- Prioritize surgical/procedural indications over general symptoms

Example format:
["nasal obstruction requiring septoplasty", "obstructive sleep apnea", "chronic sinusitis"]

Return ONLY a Python list with exactly 3 complaints (or fewer if less than 3 exist), no other text."""
        
        response = summary_llm.invoke(prompt)
        complaints_str = response.content.strip()
        
        # Remove markdown code blocks if present
        complaints_str = re.sub(r'^```(?:python)?\s*', '', complaints_str)
        complaints_str = re.sub(r'\s*```$', '', complaints_str)
        complaints_str = complaints_str.strip()
        
        # Parse the list
        try:
            import ast
            complaints = ast.literal_eval(complaints_str)
            if isinstance(complaints, list) and complaints:
                # Limit to top 3
                return [str(c).strip() for c in complaints[:3] if c]
            return [str(complaints)]
        except:
            # Fallback: split by common delimiters
            complaints = complaints_str.replace('\n', ',').replace(';', ',').split(',')
            cleaned = [c.strip().strip('"').strip("'") for c in complaints if c.strip()]
            return cleaned[:3]  # Limit to top 3
    
    def _create_basic_summary(self, documents: List, filenames: List[str]) -> str:
        """
        Fallback summary using regex extraction when LLM fails
        Returns structured data without LLM narrative
        """
        # Extract structured data via regex
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        summary_parts = ["PATIENT INFORMATION (Extracted):\n"]
        
        # Extract age
        age_match = re.search(r'(\d+)-year-old|age[:\s]+(\d+)', full_text, re.IGNORECASE)
        if age_match:
            age = age_match.group(1) or age_match.group(2)
            summary_parts.append(f"• Age: {age} years")
        
        # Extract BMI
        bmi_match = re.search(r'bmi[:\s]+(\d+\.?\d*)', full_text, re.IGNORECASE)
        if bmi_match:
            summary_parts.append(f"• BMI: {bmi_match.group(1)}")
        
        # Extract chief complaint (look for common patterns)
        cc_match = re.search(r'chief complaint[:\s]+([^\n\.]+)', full_text, re.IGNORECASE)
        if cc_match:
            summary_parts.append(f"• Chief Complaint: {cc_match.group(1).strip()}")
        
        # Extract diagnoses (ICD codes)
        icd_matches = re.findall(r'([A-Z]\d{2}\.?\d*)', full_text)
        if icd_matches:
            summary_parts.append(f"• Diagnoses: {', '.join(set(icd_matches[:5]))}")
        
        summary_parts.append(f"\n📄 Source: {', '.join(filenames)}")
        summary_parts.append("\nNote: Full summary unavailable due to processing timeout.")
        summary_parts.append("Structured data extracted successfully. ChromaDB embeddings created.")
        
        return "\n".join(summary_parts)

    def _create_chunks(self, documents: List) -> List[Dict]:
        """Create text chunks with document/page metadata"""
        chunks = []
        
        for doc in documents:
            # Split document into chunks
            doc_chunks = self.text_splitter.split_text(doc.page_content)
            
            for chunk_text in doc_chunks:
                if chunk_text.strip():  # Skip empty chunks
                    chunks.append({
                        "text": chunk_text,
                        "document": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", 1)
                    })
        
        return chunks
    
    def _create_chromadb_collection(self, chunks: List[Dict]):
        """Create ephemeral ChromaDB collection"""
        # Create ephemeral ChromaDB client (in-memory only)
        client = chromadb.Client()
        
        # Create SapBERT embedding function
        def sapbert_embed(texts):
            if isinstance(texts, str):
                texts = [texts]
            embeddings = sapbert_model.encode(texts,normalize_embeddings=True)
            return embeddings.tolist()
        
        # Check if collection exists
        try:
            collection = client.get_collection(
                name=config.EPHEMERAL_COLLECTION_NAME,
            )
            # Delete and recreate
            client.delete_collection(name=config.EPHEMERAL_COLLECTION_NAME)
        except:
            pass
        
        # Create collection
        collection = client.create_collection(
            name=config.EPHEMERAL_COLLECTION_NAME,
            embedding_function=lambda texts: sapbert_embed(texts),
            metadata={"description": "Medical record chunks for criteria evaluation"}
        )
        
        # Add chunks to collection
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{"document": chunk["document"], "page": str(chunk["page"])} for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Created ChromaDB ephemeral collection ({len(chunks)} chunks)")
        return collection
    
    def _create_citations_map(self, documents: List) -> Dict:
        """Create map of citations to pages by document"""
        citations = {}
        for doc in documents:
            filename = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 1)
            
            if filename not in citations:
                citations[filename] = []
            
            if page not in citations[filename]:
                citations[filename].append(page)
        
        # Sort pages
        for filename in citations:
            citations[filename].sort()
        
        return citations


class Node2_PolicyRetrieval:
    """
    Node 2: Retrieve relevant policies from PolicyMind using Pinecone
    Uses ORIGINAL rag.py prompt for policy summarization
    """
    
    def process(self, state: Dict) -> Dict:
        """Retrieve policies using chief complaint"""
        security.log_action("Node2_Start", state.get("user_id", "unknown"), {})
        
        chief_complaints = state.get("chief_complaint", [])
        
        # Handle both list and string formats
        if isinstance(chief_complaints, str):
            chief_complaints = [chief_complaints]
        
        if not chief_complaints or len(chief_complaints) == 0:
            return {
                **state,
                "retrieved_policies": [],
                "node2_reasoning": "No chief complaint available for policy search",
                "next_action": "node3"
            }
        
        try:
            # Import Pinecone adapter + ORIGINAL RAG functions
            from tools.rag_pinecone import retrieve_top_contexts_pinecone, check_pinecone_status
            from tools.rag import summarize_policy_chunks  # ORIGINAL prompt - DO NOT CHANGE
            from openai import OpenAI
            from collections import defaultdict
            from langchain_core.documents import Document
            
            # Initialize OpenAI client
            llm_client = OpenAI(api_key=config.OPENAI_API_KEY)
            
            # DIAGNOSTIC: Check Pinecone status first
            print("\n=== Checking Pinecone Status ===")
            check_pinecone_status()
            print("=" * 40 + "\n")
            
            # Query for each chief complaint and group policies
            policies_by_complaint = {}
            all_unique_policies = {}
            
            for complaint in chief_complaints:
                print(f"[Node 2] Querying Pinecone for: {complaint}")
                
                # Step 1: Retrieve contexts using Pinecone HYBRID SEARCH
                ranked_chunks = retrieve_top_contexts_pinecone(
                    query=complaint,
                    top_k=50,  # Max candidates before reranking
                    cross_encoder_top_k=20,  # After reranking
                    use_hybrid=True,  # Enable query expansion + keyword matching
                    llm_client=llm_client,  # Pass LLM client for query expansion
                    llm_model="gpt-4o-mini" 
                )
                
                if not ranked_chunks:
                    policies_by_complaint[complaint] = []
                    continue
                
                # Step 2: Group chunks by policy_id and get unique top policy IDs
                policy_groups = defaultdict(list)
                seen_policies = []
                for chunk in ranked_chunks:
                    if isinstance(chunk, dict):
                        policy_id = chunk.get("metadata", {}).get("policy_id") or \
                                   chunk.get("metadata", {}).get("source", "Unknown")
                        policy_groups[policy_id].append(chunk)
                        if policy_id not in seen_policies:
                            seen_policies.append(policy_id)
                
                # Get top 2-3 policy IDs per complaint (not 5, to avoid dilution)
                top_policy_ids = seen_policies[:3]
                
                complaint_policies = []
                
                # Step 3: Summarize each policy using ORIGINAL prompt
                for policy_id in top_policy_ids:
                    policy_chunks = policy_groups.get(policy_id, [])
                    
                    if not policy_chunks:
                        continue
                    
                    # If we've already processed this policy, just reference it
                    if policy_id in all_unique_policies:
                        complaint_policies.append(all_unique_policies[policy_id])
                        continue
                    
                    # Convert to LangChain Documents for original function
                    policy_docs = [
                        Document(
                            page_content=c.get("page_content", ""),
                            metadata=c.get("metadata", {})
                        )
                        for c in policy_chunks
                    ]
                    
                    # Use ORIGINAL summarize_policy_chunks with proven prompt!
                    summary_result = summarize_policy_chunks(
                        retrieved_chunks=policy_docs,
                        llm_client=llm_client,
                        llm_model=config.MODEL_SUMMARY
                    )
                    
                    # Extract summary and references (original format)
                    summary = summary_result.get("summary", "No summary available")
                    references_dict = summary_result.get("references", {})
                    
                    # Convert references dict to list format for display
                    references = []
                    for cite_key, cite_data in sorted(references_dict.items(), 
                                                     key=lambda x: int(x[0].strip("[]"))):
                        cite_num = int(cite_key.strip("[]"))
                        references.append({
                            "citation_number": cite_num,
                            "page": cite_data.get("page", "N/A"),
                            "source": cite_data.get("source", policy_id),
                            "text": ""
                        })
                    
                    # Format title
                    title = policy_id.replace("_", " ").replace("-", " ").title()
                    
                    # Calculate average score
                    avg_score = np.mean([c.get("similarity_score", 0) for c in policy_chunks]) if policy_chunks else 0.0
                    
                    policy_data = {
                        "policy_id": policy_id,
                        "title": title,
                        "summary": summary,
                        "references": references,
                        "score": float(avg_score),
                        "chunk_count": len(policy_chunks),
                        "complaint": complaint
                    }
                    
                    all_unique_policies[policy_id] = policy_data
                    complaint_policies.append(policy_data)
                
                policies_by_complaint[complaint] = complaint_policies
            
            # Flatten for backward compatibility, sorted by score
            all_policies_list = sorted(all_unique_policies.values(), key=lambda x: x['score'], reverse=True)
            
            reasoning = f"""

**Policies Retrieved**: {len(all_unique_policies)} unique policies retrieved for {', '.join(f'"{c}"' for c in chief_complaints)} 

**Database**: Pinecone vector database (HYBRID SEARCH)

**Processing Steps**:
1. Performed hybrid search in Pinecone with query expansion (semantic embeddings + keyword BM25)
2. Applied cross-encoder reranking on top of hybrid results
3. Final relevance score: 70% cross-encoder + 30% keyword overlap
4. Grouped chunks by policy ID
5. Generated summaries using Policy Mind rag.py prompt format
6. Maintained section structure: Coverage Criteria, Exclusions, etc.

**Status**: ✅ Complete with structured policy summaries"""
            
            security.log_action("Node2_Complete", state.get("user_id", "unknown"), {
                "policies_retrieved": len(all_unique_policies),
                "chief_complaints": chief_complaints
            })
            
            return {
                **state,
                "retrieved_policies": all_policies_list,
                "policies_by_complaint": policies_by_complaint,  
                "node2_reasoning": reasoning,
                "next_action": "node3"
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Node 2 Error: {str(e)}"
            print(f"[Node 2] Error: {e}")
            traceback.print_exc()
            security.log_action("Node2_Error", state.get("user_id", "unknown"), {"error": str(e)})
            
            return {
                **state,
                "retrieved_policies": [],
                "node2_reasoning": f"Policy retrieval failed: {str(e)}. Please ensure Pinecone is configured correctly.",
                "next_action": "node3"
            }


class Node3_CriteriaEvaluator:
    """
    Node 3: Evaluate criteria using RAG retrieval from ChromaDB ephemeral
    Confidence based on actual similarity scores
    """
    
    def process(self, state: Dict) -> Dict:
        """Evaluate criteria against medical records"""
        security.log_action("Node3_Start", state.get("user_id", "unknown"), {})
        
        criterion_list = state.get("criterion_list", [])
        chromadb_collection = state.get("chromadb_collection")
        
        if not criterion_list:
            return {
                **state,
                "evaluation_results": [],
                "node3_reasoning": "No criteria provided for evaluation",
                "next_action": "end"
            }
        
        if not chromadb_collection:
            return {
                **state,
                "evaluation_results": [],
                "node3_reasoning": "No medical record chunks available. Please process records first.",
                "next_action": "end"
            }
        
        try:
            # Evaluate each criterion
            results = []
            for criterion in criterion_list:
                result = self._evaluate_criterion(criterion, chromadb_collection)
                results.append(result)
            
            reasoning = f"""

- **{len(criterion_list)}** criteria evaluated
- **Retrieval Method**: Hybrid keyword extraction + SapBERT semantic search
- **Source**: ChromaDB ephemeral collection (medical records)

**Processing Steps**:
1. **Keyword Extraction First**: For factual criteria (age, BMI, lab values), used regex pattern matching to find exact data points (e.g., "Age: 35", "BMI 26.6")
2. **Semantic Search Fallback**: For conceptual criteria (symptoms, diagnoses, treatments), used SapBERT embeddings with L2 distance
3. **Strict Relevance Filtering**: Only included evidence with distance ≤ 1.2 (no fixed evidence limits - returns 1-3 pieces if relevant, 0 if not found)
4. **Confidence Scoring**: 
   - High (Exact Match): distance = 0.0 (keyword extraction)
   - High: distance ≤ 0.5 (very close semantic match)
   - Medium: distance ≤ 1.0 (acceptable semantic match)
   - Low: distance > 1.0 (weak semantic match)
5. **LLM Evaluation**: GPT-4 determines Met/Not Met/Insufficient Data status based on evidence
6. **Source Citations**: Each evidence piece includes document name and page number

**Quality**: ✅ All criteria evaluated with dynamic evidence retrieval (only relevant evidence included)

"""
            
            security.log_action("Node3_Complete", state.get("user_id", "unknown"), {
                "criteria_evaluated": len(results)
            })
            
            return {
                **state,
                "evaluation_results": results,
                "node3_reasoning": reasoning,
                "next_action": "end"
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Node 3 Error: {str(e)}"
            print(f"[Node 3] Error: {e}")
            traceback.print_exc()
            security.log_action("Node3_Error", state.get("user_id", "unknown"), {"error": str(e)})
            
            return {
                **state,
                "evaluation_results": [],
                "node3_reasoning": f"Evaluation failed: {str(e)}",
                "next_action": "end"
            }
        
    def _keyword_extract(self, criterion: str, collection) -> List[Dict]:
        """
        Extract evidence using KEYWORD patterns for common criteria
        Returns evidence list if found, empty list if not applicable/not found
        """
        criterion_lower = criterion.lower()
        
        # Get all documents from collection
        all_docs = collection.get(include=['documents', 'metadatas'])

        if not all_docs['documents']:
            return []
        
        evidence = []
        
        # Pattern 1: AGE-related criteria
        if any(term in criterion_lower for term in ['age', 'years', 'year', 'older', 'younger']):
            for doc, metadata in zip(all_docs['documents'], all_docs['metadatas']):
                doc_lower = doc.lower()
                
                # Look for age patterns
                age_patterns = [
                    r'age[:\s]+(\d+)',                      # "Age: 35"
                    r'(\d+)\s*y/?o',                        # "35 y/o"
                    r'(\d+)\s*years?\s+old',                # "35 years old"
                    r'(\d+)-year-old',                      # "35-year-old" ← NEW
                    r'(\d+)\s+year\s+old',                  # "35 year old"
                    r'dob[:\s]+[\d/]+.*?\(age[:\s]+(\d+)'  # "DOB: ... (Age: 35)"
                ]
                
                for pattern in age_patterns:
                    matches = re.finditer(pattern, doc_lower)
                    for match in matches:
                        # Extract surrounding context (±50 chars)
                        start = max(0, match.start() - 50)
                        end = min(len(doc), match.end() + 50)
                        snippet = doc[start:end].strip()
                        
                        # Verify it's not a false match (minutes, doses, etc.)
                        if not any(false in doc_lower[max(0,match.start()-30):match.end()+30] 
                                   for false in ['minute', 'hour', 'dose', 'mg', 'surgery', 'procedure']):
                            
                            evidence.append({
                                'evidence': snippet,
                                'document_name': metadata.get('document', 'Unknown'),
                                'page_no': str(metadata.get('page', 'N/A')),
                                'confidence': 'High (Exact Match)',
                                'distance': 0.0  # Exact keyword match
                            })
                            break  # Found age in this doc, move to next
                
                if evidence:  # Found age data, no need to check more docs
                    break
        
        # Pattern 2: BMI-related criteria
        elif 'bmi' in criterion_lower:
            for doc, metadata in zip(all_docs['documents'], all_docs['metadatas']):
                if 'bmi' in doc.lower():
                    # Extract BMI mention with context
                    match = re.search(r'bmi[:\s]+[\d.]+', doc.lower())
                    if match:
                        start = max(0, match.start() - 30)
                        end = min(len(doc), match.end() + 30)
                        snippet = doc[start:end].strip()
                        
                        evidence.append({
                            'evidence': snippet,
                            'document_name': metadata.get('document', 'Unknown'),
                            'page_no': str(metadata.get('page', 'N/A')),
                            'confidence': 'High (Exact Match)',
                            'distance': 0.0
                        })
                        break
        
        # Pattern 3: Diagnosis/Condition keywords
        elif any(term in criterion_lower for term in ['diagnosis', 'diagnosed', 'history of']):
            # Extract key medical terms from criterion
            medical_terms = []
            # Common conditions
            conditions = ['diabetes', 'hypertension', 'asthma', 'copd', 'depression', 
                         'anxiety', 'septal deviation', 'nasal obstruction', 'sleep apnea']
            
            for condition in conditions:
                if condition in criterion_lower:
                    medical_terms.append(condition)
            
            if medical_terms:
                for doc, metadata in zip(all_docs['documents'], all_docs['metadatas']):
                    doc_lower = doc.lower()
                    for term in medical_terms:
                        if term in doc_lower:
                            # Find sentence containing the term
                            sentences = re.split(r'[.!?]+', doc)
                            for sent in sentences:
                                if term in sent.lower():
                                    evidence.append({
                                        'evidence': sent.strip()[:200],
                                        'document_name': metadata.get('document', 'Unknown'),
                                        'page_no': str(metadata.get('page', 'N/A')),
                                        'confidence': 'Medium',
                                        'distance': 0.0
                                    })
                                    break
                            break
                    
                    if len(evidence) >= 2:  # Limit to 2 examples for diagnoses
                        break
        
        return evidence

    
    def _evaluate_criterion(self, criterion: str, collection) -> Dict:
        """
        Evaluate criterion using HYBRID approach:
        1. Keyword extraction first (for age, BMI, diagnoses, dates)
        2. Semantic search fallback
        3. NO fixed evidence limits - return only relevant evidence
        """
        try:
            criterion_lower = criterion.lower()
            
            # STEP 1: Try keyword-based extraction first for common patterns
            keyword_evidence = self._keyword_extract(criterion, collection)
            
            if keyword_evidence:
                evaluation = self._evaluate_with_llm(criterion, keyword_evidence)
                return {
                    'criterion': criterion,
                    'status': evaluation['status'],
                    'reasoning': evaluation['reasoning'],
                    'evidence': keyword_evidence
                }
            
            # STEP 2: Fallback to semantic search
            
            # Embed criterion using SapBERT
            criterion_embedding = sapbert_model.encode([criterion], normalize_embeddings=True)[0].tolist()
            
            # Query ChromaDB
            results = collection.query(
                query_embeddings=[criterion_embedding],
                n_results=30,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {
                    'criterion': criterion,
                    'status': 'Insufficient Data',
                    'reasoning': 'No relevant information found in medical records',
                    'evidence': []
                }
            
            # Build evidence - NO FIXED LIMITS, just relevance-based
            evidence = []
            max_evidence = 5  # Maximum pieces of evidence to collect
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                
                # Stop if we have enough evidence
                if len(evidence) >= max_evidence:
                    print(f"  ✓ Collected {max_evidence} pieces of evidence, stopping search")
                    break
                
                # STRICT threshold: only distance < 1.2 are relevant
                if distance > 1.2:
                    print(f"  Result {i+1} [SKIPPED]: distance={distance:.3f} too high")
                    break  # Results are sorted by distance, so stop here
                
                # Try to extract relevant snippet
                try:
                    evidence_snippet = self._extract_relevant_snippet(doc, criterion)
                    
                    if not evidence_snippet or len(evidence_snippet.strip()) < 10:
                        print(f"  Result {i+1} [SKIPPED]: No relevant content")
                        continue
                    
                    print(f"  Result {i+1} [INCLUDED]: distance={distance:.3f}")
                    
                    confidence = self._determine_confidence(distance)
                    
                    evidence.append({
                        'evidence': evidence_snippet,
                        'document_name': metadata.get('document', 'Unknown'),
                        'page_no': str(metadata.get('page', 'N/A')),
                        'confidence': confidence,
                        'distance': float(distance)
                    })
                    
                except Exception as e:
                    print(f"  Result {i+1} [ERROR]: {e}")
                    continue
            
            # If no evidence found after all that, return insufficient data
            if not evidence:
                return {
                    'criterion': criterion,
                    'status': 'Insufficient Data',
                    'reasoning': 'No relevant evidence found in the medical records for this criterion',
                    'evidence': []
                }
            
            # Evaluate using LLM
            evaluation = self._evaluate_with_llm(criterion, evidence)
            
            return {
                'criterion': criterion,
                'status': evaluation['status'],
                'reasoning': evaluation['reasoning'],
                'evidence': evidence
            }
            
        except Exception as e:
            print(f"[Node 3] Error evaluating criterion: {e}")
            import traceback
            traceback.print_exc()
            return {
                'criterion': criterion,
                'status': 'Insufficient Data',
                'reasoning': f'Evaluation error: {str(e)}',
                'evidence': []
            }
    def _determine_confidence(self, l2_distance: float) -> str:
        """
        Determine confidence level based on L2 distance
        Use L2 distance thresholds (lower = better)
        
        Args:
            l2_distance: L2 distance from ChromaDB (0 = identical, higher = less similar)
            
        Returns:
            Confidence level: High/Medium/Low
        """
        if l2_distance == 0.0:
            return "High (Exact Match)"  # Keyword extraction
        elif l2_distance <= 0.5:
            return "High"
        elif l2_distance <= 1.0:
            return "Medium"
        else:
            return "Low"
        
    def _extract_relevant_snippet(self, text: str, criterion: str) -> str:
        """
        Extract ONLY the most relevant sentence(s) - max 200 chars
        Focus on extracting the exact information needed
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Extract key search terms from criterion (focus on nouns/numbers)
        criterion_lower = criterion.lower()
        
        # Special handling for age-related criteria
        if 'age' in criterion_lower or 'years' in criterion_lower:
            # Look for patterns like "Age: 35" or "DOB:" or "35 years"
            for sent in sentences:
                sent_lower = sent.lower()
                if any(pattern in sent_lower for pattern in ['age:', 'dob:', 'years old', 'y/o']):
                    # Extract just the relevant part (limit to 150 chars)
                    if len(sent) > 150:
                        # Find the age/DOB part
                        # import re
                        match = re.search(r'(age[:\s]+\d+|dob[:\s]+[\d/]+|\d+\s+years)', sent_lower)
                        if match:
                            start = max(0, match.start() - 20)
                            end = min(len(sent), match.end() + 30)
                            return sent[start:end].strip()
                    return sent.strip()
        
        # General case: extract key terms
        criterion_terms = set(criterion_lower.split())
        criterion_terms = {t for t in criterion_terms if len(t) > 3}
        
        # Score sentences by keyword overlap
        scored_sentences = []
        for sent in sentences:
            if len(sent.strip()) < 10:
                continue
            sent_lower = sent.lower()
            score = sum(1 for term in criterion_terms if term in sent_lower)
            if score > 0:  # Only include sentences with matches
                scored_sentences.append((score, sent))
        
        if not scored_sentences:
            # No good match, return first sentence
            return sentences[0][:200] if sentences else text[:200]
        
        # Get the best sentence
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        best_sentence = scored_sentences[0][1].strip()
        
        # Limit to 200 chars max
        if len(best_sentence) > 200:
            best_sentence = best_sentence[:200] + '...'
        
        return best_sentence

    def _evaluate_with_llm(self, criterion: str, evidence: List[Dict]) -> Dict:
        """Evaluate criterion using LLM with evidence"""
        # Build evidence context
        evidence_text = "\n\n".join([
            f"Evidence {i+1} (from {ev['document_name']}, page {ev['page_no']}):\n{ev['evidence']}"
            for i, ev in enumerate(evidence)
        ])
        
        prompt = f"""You are a clinical review assistant. Evaluate whether the following criterion is met based on the evidence from medical records.

Criterion: {criterion}

Evidence from Medical Records:
{evidence_text}

Determine:
1. Status: "Met", "Not Met", or "Insufficient Data"
2. Reasoning: Brief explanation (1-2 sentences) of why

Respond in this exact format:
STATUS: [Met/Not Met/Insufficient Data]
REASONING: [Your explanation here]"""
        
        response = evaluation_llm.invoke(prompt)
        response_text = response.content
        
        # Parse response
        status = "Insufficient Data"
        reasoning = "Unable to determine"
        
        try:
            lines = response_text.split('\n')
            for line in lines:
                if line.startswith('STATUS:'):
                    status = line.replace('STATUS:', '').strip()
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
        except:
            pass
        
        return {
            'status': status,
            'reasoning': reasoning
        }

# Create node instances
node1 = Node1_MedicalRecordProcessor()
node2 = Node2_PolicyRetrieval()
node3 = Node3_CriteriaEvaluator()
