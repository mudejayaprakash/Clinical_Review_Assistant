"""
Clinical Review Assistant - Streamlit Application
"""
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

from agents.auth import auth
from agents.agent import agent
from agents.security import security
# import agents.config as config
from agents.config import PAGE_TITLE, PAGE_ICON, LAYOUT

# Session file for persistent login
SESSION_FILE = "session.json"

# Reusable styled section header
        # background-color:#c1e2ea;

def section_header(title):
    st.markdown(f"""
    <div style="
        display:flex;
        align-items:center;
        background: linear-gradient(90deg, #c7e8eb 0%, #a4dce0 100%);
        border-left:8px solid #005e63;
        padding:10px 14px;
        margin:10px 0;
        border-radius:6px;">
        <span style='color:#007a80;
                     font-size:1.4rem;
                     font-weight:700;'>
            {title}
        </span>
    </div>
    """, unsafe_allow_html=True)


def save_session(user_id):
    """Save user session to file"""
    with open(SESSION_FILE, 'w') as f:
        json.dump({
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }, f)

def load_session():
    """Load user session from file"""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def clear_session():
    """Clear saved session"""
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT
)

# Initialize session state
if "authenticated" not in st.session_state:
    # Check for saved session
    saved_session = load_session()
    if saved_session:
        st.session_state.authenticated = True
        st.session_state.user_id = saved_session["user_id"]
    else:
        st.session_state.authenticated = False
        st.session_state.user_id = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "file_selections" not in st.session_state:
    st.session_state.file_selections = {}
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "node1_complete" not in st.session_state:
    st.session_state.node1_complete = False
if "node2_complete" not in st.session_state:
    st.session_state.node2_complete = False


def main():
    """Main application logic"""
    # Check authentication
    if not st.session_state.authenticated:
        auth.login_page()
        return
    
    # Main application
    render_main_app()


def render_main_app():
    """Render main application interface"""
    
    # Reduce spacing around dividers
    st.markdown("""
        <style>
        hr {
            margin-top: 0.5rem !important; margin-bottom: 0.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar with user info and controls
    with st.sidebar:
        st.markdown(f"<h1 style='font-size: 2rem;'>👤  {st.session_state.user_id.title()}</h1>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 1.5px solid #007a80ff; margin: 20px 0;'>", unsafe_allow_html=True)


        if st.button("🔄 Reset Results", width='stretch', type="secondary"):
            # Keep uploaded files, but clear all processing results
            st.session_state.agent_state = None
            st.session_state.processing_complete = False
            st.session_state.node1_complete = False
            st.session_state.node2_complete = False
            st.session_state.evaluation_results = None
            st.session_state.file_hash = None 
            st.session_state.just_reset = True  # ← ADD THIS FLAG
            
            st.success("✅ Processing results cleared. Files remain uploaded - select and summarize again.")
            st.rerun()

        # Clear All Files button - removes everything including files
        if st.button("🗑️ Clear All Files", width='stretch', type="secondary"):
            current_key = st.session_state.get("uploader_key", 0)
            st.session_state.clear()
            st.session_state.uploader_key = current_key + 1
            st.success("✅ All files and results cleared.")
            st.rerun()

        if st.button("🔓 Logout", width='stretch', type="primary"):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.uploaded_files = []
            st.session_state.agent_state = None
            st.session_state.processing_complete = False
            clear_session()
            st.rerun()
    
    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #4b79a1 0%, #004e80 100%);
        border-radius: 10px;
        padding: 10px 25px;
        margin-bottom: 2rem;
        color: white;
    ">
        <h1 style="margin: 0; font-size: 2.7rem;"> 📋 Clinical Review Assistant</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.4rem; opacity: 0.9;">
            AI-Powered Medical Utilization Review
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Single unified workflow
    render_unified_workflow()


def render_unified_workflow():
    """Render single unified tab with sequential workflow"""
    
    # SECTION 1: File Upload
    section_header("Upload Medical Records")
    
    uploaded_files = st.file_uploader(
        "Please upload one or more PDF medical records to process.",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}"
    )
    # Check if we just reset (flag from reset button)
    just_reset = st.session_state.get("just_reset", False)
    if just_reset:
        # Clear the flag immediately
        st.session_state.just_reset = False
         
    elif uploaded_files is not None and len(uploaded_files) > 0:
        # Get current filenames in uploader
        current_filenames = [file.name for file in uploaded_files]
        
        # Remove files that were deleted from uploader (via X button)
        st.session_state.uploaded_files = [
            f for f in st.session_state.uploaded_files 
            if f["filename"] in current_filenames
        ]
        
        # Remove from selections too
        for filename in list(st.session_state.file_selections.keys()):
            if filename not in current_filenames:
                del st.session_state.file_selections[filename]
        
        # Add new files
        for file in uploaded_files:
            # Check if file already exists
            if not any(f["filename"] == file.name for f in st.session_state.uploaded_files):
                file_bytes = file.read()
                
                # Validate PDF
                is_valid, error_msg = security.validate_pdf(file_bytes)
                if is_valid:
                    st.session_state.uploaded_files.append({
                        "filename": file.name,
                        "content": file_bytes,
                        "selected": True
                    })
                    # Initialize selection state
                    if file.name not in st.session_state.file_selections:
                        st.session_state.file_selections[file.name] = True
                else:
                    st.error(f"{file.name}: {error_msg}")
                    
    elif uploaded_files is not None and len(uploaded_files) == 0:
        # Widget is empty but session has files = ALL removed via X button
        st.session_state.uploaded_files = []
        st.session_state.file_selections = {}
        st.session_state.agent_state = None
        st.session_state.processing_complete = False
        st.session_state.node1_complete = False
        st.session_state.node2_complete = False
        st.session_state.evaluation_results = None
        st.session_state.file_hash = None
    
    # Display uploaded files with selection 
    if st.session_state.uploaded_files:
        st.markdown("#### Select Records to process:")

        # Display files with selection checkboxes
        selected_count = 0
        for i, file_info in enumerate(st.session_state.uploaded_files):
            selected = st.checkbox(
                file_info["filename"],
                value=st.session_state.file_selections.get(file_info["filename"], True),
                key=f"select_{file_info['filename']}_{i}"
            )
            st.session_state.file_selections[file_info["filename"]] = selected
            file_info["selected"] = selected
            
            if selected:
                selected_count += 1
        
        st.caption(f"{selected_count} file(s) selected for processing")
                
        # # SECTION 2: Process Records
        
        if st.button("Summarize and Analyze Records", type="primary"):
            if selected_count == 0:
                st.error("Please select at least one file to process")
            else:
                # Calculate hash of current file selection
                import hashlib
                selected_files = [f['filename'] for f in st.session_state.uploaded_files 
                                 if st.session_state.file_selections.get(f['filename'], False)]
                selected_files.sort()  # Ensure consistent ordering
                current_hash = hashlib.md5(str(selected_files).encode()).hexdigest()
                
                # Check if same files already processed
                if (st.session_state.file_hash == current_hash and 
                    st.session_state.processing_complete and 
                    st.session_state.agent_state):
                    st.info("**Using cached results** - Same documents already processed. Click 'Reset' to reprocess.")
                else:
                    # Files changed or first run - clear old state and process
                    if st.session_state.file_hash != current_hash:
                        # Different files selected - clear previous results
                        st.session_state.agent_state = None
                        st.session_state.processing_complete = False
                        st.session_state.node1_complete = False
                        st.session_state.node2_complete = False
                        st.session_state.evaluation_results = None
                    
                    # Store current hash
                    st.session_state.file_hash = current_hash
                process_records()
        
        # SECTION 3: Display Results (if Node 1 is complete)
        if st.session_state.node1_complete and st.session_state.agent_state:
            display_processing_results()

        # Auto-trigger Node 2 if Node 1 is complete but Node 2 isn't
        if st.session_state.node1_complete and not st.session_state.node2_complete:
            process_records()

def process_records():
    """
    Process medical records progressively: Node 1 → Node 2
    Shows results incrementally for better UX
    """
    # STAGE 1: Process Node 1 (if not done)
    if not st.session_state.node1_complete:
        progress_placeholder = st.empty()
        
        with progress_placeholder:
            st.info("🔄 Step 1/2: Processing medical records...")
        
        try:
            # Run Node 1 only
            state = agent.process_node1_only(
                medical_records=st.session_state.uploaded_files,
                user_id=st.session_state.user_id
            )
            
            # Check for errors
            if state.get("errors"):
                progress_placeholder.empty()
                for error in state["errors"]:
                    st.error(f"{error}")
                return
            
            # Store state and mark Node 1 complete
            st.session_state.agent_state = state
            st.session_state.node1_complete = True
            
            progress_placeholder.empty()
            st.success("✅ Medical records processed!")
            
            # Rerun to display Node 1 results
            st.rerun()
            
        except Exception as e:
            st.error(f"Error in Node 1: {str(e)}")
            return
    
    # STAGE 2: Process Node 2 (if Node 1 done but Node 2 not done)
    if st.session_state.node1_complete and not st.session_state.node2_complete:
        progress_placeholder = st.empty()
        
        with progress_placeholder:
            st.info("🔄 Step 2/2: Retrieving relevant policies from Pinecone...")
        
        try:
            # Run Node 2 only
            state = agent.process_node2_only(
                state=st.session_state.agent_state,
                user_id=st.session_state.user_id
            )
            
            # Check for errors
            if state.get("errors"):
                progress_placeholder.empty()
                for error in state["errors"]:
                    st.error(f"{error}")
                return
            
            # Update state and mark Node 2 complete
            st.session_state.agent_state = state
            st.session_state.node2_complete = True
            st.session_state.processing_complete = True
            
            progress_placeholder.empty()
            st.success("✅ Policies retrieved!")
            
            # Rerun to display Node 2 results
            st.rerun()
            
        except Exception as e:
            st.error(f"Error in Node 2: {str(e)}")
            return

def display_processing_results():
    """
    Display results from Node 1 and Node 2
    """
    state = st.session_state.agent_state
    if not state:
        return
    
    # Node 1 Results: Summary
    section_header(" Medical Record Summary")    
    # Display summary
    summary = state.get("summary", "No summary available")
    if "**Citations by Document:**" in summary:
        summary = summary.split("**Citations by Document:**")[0].strip()
    
    st.markdown(summary)
    
    # Chief Complaints Section
    section_header("Chief Complaints")
    chief_complaints = state.get('chief_complaint', [])
    
    if isinstance(chief_complaints, str):
        chief_complaints = [chief_complaints]
    
    if chief_complaints:
        for idx, complaint in enumerate(chief_complaints, 1):
            st.markdown(
                f"""<p style='font-size: 1.1rem; font-weight: 400; line-height: 1; 
                margin: 4px 4px 20px 4px;'>
                <b>{idx}.</b> {complaint.capitalize()}
                </p>""",
                unsafe_allow_html=True
            )

    else:
        st.caption("Not identified")

    st.markdown("---")
    # Display Node 1 reasoning (collapsible)
    with st.expander("🧠 Node 1 Reasoning", expanded=False):
        st.markdown(state.get("node1_reasoning", "No reasoning available"))
      
    # Anchor for auto-scroll AFTER Node 1 results
    st.markdown('<div id="node2-anchor"></div>', unsafe_allow_html=True)
    
    # Auto-scroll to this point (where Node 2 processing message will appear)
    if not st.session_state.node2_complete:
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
                setTimeout(function() {
                    const anchor = window.parent.document.getElementById('node2-anchor');
                    if (anchor) {
                        anchor.scrollIntoView({behavior: 'smooth', block: 'start'});
                    }
                }, 200);
            </script>
            """,
            height=0,
        )

    # Node 2 Results: Policies grouped by Chief Complaint. Show only if Node 2 is complete
    if st.session_state.node2_complete:
        section_header("Relevant Medical Policies by Chief Complaint")

        policies = state.get("retrieved_policies", [])
        policies_by_complaint = state.get("policies_by_complaint", {})
        
        if policies and policies_by_complaint and chief_complaints:
            st.caption(f"Found **{len(policies)}** unique relevant policy documents overall.")            
            # Each chief complaint is a collapsible section
            for idx, complaint in enumerate(chief_complaints):
                complaint_policies = policies_by_complaint.get(complaint, [])
                if not complaint_policies:
                    continue

                with st.expander(
                    f"**{complaint.capitalize()}** ({len(complaint_policies)} policies)",
                    expanded=False
                ):
                    # Display policies under this complaint
                    for i, policy in enumerate(complaint_policies):
                        policy_id = policy.get('policy_id', 'Unknown')
                        score = policy.get('score', 0.0)
                        chunk_count = policy.get('chunk_count', 0)
                        
                        # Policy expander - CSS resets this to default white
                        with st.expander(f"📄 {policy_id}", expanded=False):
                            st.caption(f"📊 {chunk_count} sections analyzed | Relevance: {score:.3f}")
                            summary = policy.get("summary", "No summary available")
                            st.markdown(summary)

                            # References section
                            references = policy.get("references", [])
                            if references:
                                with st.expander("References", expanded=False):
                                    ref_list = ", ".join([
                                        f"**[{ref['citation_number']}]** Page {int(float(ref['page']))}" 
                                        for ref in references
                                    ])
                                    st.markdown(ref_list)

        else:
            st.warning("⚠️ No policies retrieved.")

        st.markdown("---")

        # Node 2 reasoning - will also have gray/blue styling from CSS
        with st.expander("🧠 Node 2 Reasoning", expanded=False):
            st.markdown(state.get("node2_reasoning", "No reasoning available"))
    
    
    # Node 3: Criteria Evaluation. Show only if Node 2 is complete
    if st.session_state.node2_complete:
        section_header("Criteria Evaluation")
        
        st.markdown("**Enter criteria to evaluate (one per line):**")
        
        criteria_text = st.text_area(
            "Criteria",
            height=100,
            placeholder="Example:\n• Patient must have documented diagnosis\n• Treatment must be medically necessary",
            label_visibility="collapsed"
        )
        
        if st.button("Evaluate Criteria", type="primary"):
            evaluate_criteria(criteria_text)
        
        # Display evaluation results
        if st.session_state.evaluation_results:
            display_evaluation_results()


def evaluate_criteria(criteria_text: str):
    """
    Evaluate criteria using Node 3
    """
    if not criteria_text.strip():
        st.error("Please enter at least one criterion")
        return
    
    # Parse criteria (split by newlines, remove bullets)
    criterion_list = []
    for line in criteria_text.split("\n"):
        line = line.strip()
        if line:
            # Remove common bullet points
            line = line.lstrip("•-*123456789. ")
            if line:
                criterion_list.append(line)
    
    if not criterion_list:
        st.error("No valid criteria found")
        return
    
    # Progress indicator
    progress_placeholder = st.empty()
    with progress_placeholder:
        st.info(f"🔄 Evaluating {len(criterion_list)} criteria...")
    
    try:
        # Run Node 3
        state = agent.evaluate_criteria(
            state=st.session_state.agent_state,
            criterion_list=criterion_list
        )
        
        # Clear progress
        progress_placeholder.empty()
        
        # Check for errors
        if state.get("errors"):
            for error in state["errors"]:
                st.error(f"{error}")
            return
        
        # Store results
        st.session_state.evaluation_results = {
            "results": state.get("evaluation_results", []),
            "reasoning": state.get("node3_reasoning", "")
        }

        st.success(f"Evaluated {len(criterion_list)} criteria")
        st.rerun()
        
    except Exception as e:
        progress_placeholder.empty()
        st.error(f"Error: {str(e)}")
        security.log_action("App_Evaluation_Error", st.session_state.user_id, {"error": str(e)})


def display_evaluation_results():
    """
    Display Node 3 evaluation results in table format
    Confidence based on actual scores
    Text wrapping in evidence table
    """

    section_header("Evaluation Results")

    results = st.session_state.evaluation_results["results"]
    
    if not results:
        st.warning("No results available")
        return
    
    # Display each criterion result
    for i, result in enumerate(results):
        criterion = result.get("criterion", "")
        status = result.get("status", "Unknown")
        reasoning = result.get("reasoning", "")
        evidence_rows = result.get("evidence", [])
        
        # Status badge
        if status == "Met":
            status_badge = "✅ **Met**"
            status_color = "green"
        elif status == "Not Met":
            status_badge = "❌ **Not Met**"
            status_color = "red"
        else:
            status_badge = "⚠️ **Insufficient Data**"
            status_color = "orange"
        
        # Criterion heading
        st.markdown(
            f"<div style='font-size:1.2rem; font-weight:600; color:#1f2933; margin-bottom:10px;'>"
            f"Criterion {i+1}: {criterion}"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Status
        st.markdown(
            f"<p style='font-size:1rem; margin:4px 0;'>"
            f"<b>Status:</b> <span style='color:{status_color}; font-weight:600;'>{status_badge}</span>"
            f"</p>",
            unsafe_allow_html=True
        )

        # Reasoning
        st.markdown(
            f"<p style='font-size:1rem; margin:4px 0;'>"
            f"<b>Reasoning:</b> {reasoning}</p>",
            unsafe_allow_html=True
        )

        # Evidence Section
        if evidence_rows:
            st.markdown("<p style='font-size:1rem; font-weight:600; margin:6px 0;'>Evidence:</p>", unsafe_allow_html=True)

            # Filter out empty evidence, limit to top 5, and create DataFrame
            valid_evidence = []
            for e in evidence_rows:
                evidence_text = e.get('evidence', '')
                # Only include if evidence text is not empty and not just whitespace
                if evidence_text and str(evidence_text).strip():
                    valid_evidence.append(e)
                    if len(valid_evidence) >= 3:  # Stop at 3
                        break
            
            if not valid_evidence:
                st.caption("*No evidence found*")
            else:
                # Convert to DataFrame
                df = pd.DataFrame(valid_evidence)

                # Select only display columns
                display_cols = ['evidence', 'document_name', 'page_no', 'confidence']
                df = df[display_cols]
                df.columns = ['Evidence', 'Document Name', 'Page No', 'Confidence']

                # Indented data editor
                st.markdown("<div style='margin-left:15px;'>", unsafe_allow_html=True)
                st.data_editor(
                    df,
                    width='stretch',
                    hide_index=True,
                    disabled=True,
                    column_config={
                        "Evidence": st.column_config.TextColumn(width="large", help="Full text from medical record"),
                        "Document Name": st.column_config.TextColumn(width="medium"),
                        "Page No": st.column_config.TextColumn(width="small"),
                        "Confidence": st.column_config.TextColumn(width="small")
                    },
                    num_rows="fixed",
                    key=f"evidence_table_{i}"
                )
        else:
            st.caption("*No evidence found*")

        # End indented block
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.divider()
    
    # Display Node 3 reasoning (collapsible)
    with st.expander("🧠 Node 3 Reasoning", expanded=False):
        st.markdown(st.session_state.evaluation_results["reasoning"])

if __name__ == "__main__":
    main()
