"""
LangGraph Agent for Clinical Review Assistant
Single agent with 3 autonomous nodes and state management
"""
from typing import Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from chromadb import Collection

from agents.nodes import node1, node2, node3
from agents.security import security
import agents.config as config

class AgentState(TypedDict):
    """State passed between nodes"""
    # Input
    medical_records: Optional[List[Dict]]  # [{"filename": str, "content": bytes, "selected": bool}]
    criterion_list: Optional[List[str]]    # User-provided criteria to evaluate
    
    # Node 1 outputs
    summary: Optional[str]                 # Summary with inline citations grouped by document
    chief_complaint: Optional[str]         # For Node 2 policy search
    node1_reasoning: Optional[str]
    documents_processed: Optional[List[str]]
    citations_map: Optional[Dict]          # Maps citation numbers to pages by document
    record_chunks: Optional[List[Dict]]    # Chunks with metadata (document, page)
    chromadb_collection: Optional[Collection]  # Ephemeral ChromaDB collection for Node 3 retrieval
    
    # Node 2 outputs (for reference only, NOT used in Node 3)
    retrieved_policies: Optional[List[Dict]]  # Policies with grouped citations
    policies_by_complaint: Optional[Dict]
    node2_reasoning: Optional[str]
    
    # Node 3 outputs (uses RAG retrieval from chromadb_collection)
    evaluation_results: Optional[List[Dict]]  # One result per criterion with table data
    node3_reasoning: Optional[str]
    
    # Agent control
    next_action: Optional[str]
    iteration_count: int
    errors: List[str]
    user_id: Optional[str]


class ClinicalReviewAgent:
    """
    Single agent orchestrator with 3 nodes:
    - Node 1: Medical Record Processor (summarize + chunk + embed + store)
    - Node 2: PolicyMind RAG (retrieve policies for reference)
    - Node 3: Criteria Evaluator (RAG retrieval + evaluation)
    """
    
    def __init__(self):
        self.graph = self._build_graph()
        security.log_action("Agent_Initialized", "system", {})
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph with 3 nodes"""
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("node1_medical_processor", self._node1_wrapper)
        workflow.add_node("node2_policy_retrieval", self._node2_wrapper)
        workflow.add_node("node3_criteria_evaluator", self._node3_wrapper)
        
        # Set entry point
        workflow.set_entry_point("node1_medical_processor")
        
        # Add edges with conditional routing
        workflow.add_conditional_edges(
            "node1_medical_processor",
            self._route_after_node1,
            {
                "node2": "node2_policy_retrieval",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "node2_policy_retrieval",
            self._route_after_node2,
            {
                "node3": "node3_criteria_evaluator",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "node3_criteria_evaluator",
            self._route_after_node3,
            {
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _node1_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for Node 1 with iteration tracking"""
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        if state["iteration_count"] > config.MAX_AGENT_ITERATIONS:
            state["errors"].append("Maximum iterations exceeded")
            state["next_action"] = "end"
            return state
        
        return node1.process(state)
    
    def _node2_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for Node 2 with iteration tracking"""
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        if state["iteration_count"] > config.MAX_AGENT_ITERATIONS:
            state["errors"].append("Maximum iterations exceeded")
            state["next_action"] = "end"
            return state
        
        return node2.process(state)
    
    def _node3_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for Node 3 with iteration tracking"""
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        if state["iteration_count"] > config.MAX_AGENT_ITERATIONS:
            state["errors"].append("Maximum iterations exceeded")
            state["next_action"] = "end"
            return state
        
        return node3.process(state)
    
    def _route_after_node1(self, state: AgentState) -> str:
        """Route after Node 1 completes"""
        next_action = state.get("next_action", "end")
        
        if next_action == "node2":
            return "node2"
        else:
            return "end"
    
    def _route_after_node2(self, state: AgentState) -> str:
        """Route after Node 2 completes"""
        next_action = state.get("next_action", "end")
        
        if next_action == "node3":
            return "node3"
        else:
            return "end"
    
    def _route_after_node3(self, state: AgentState) -> str:
        """Route after Node 3 completes"""
        # Always end after Node 3
        return "end"
    
    def process_node1_only(
        self,
        medical_records: List[Dict],
        user_id: str
    ) -> Dict:
        """
        Process medical records through Node 1 only
        
        Args:
            medical_records: List of medical record dicts
            user_id: User ID for logging
            
        Returns:
            Final state after Node 1
        """
        security.log_action("Agent_Process_Records_Start", user_id, {
            "num_records": len(medical_records)
        })
        
        # Initialize state
        initial_state = AgentState(
            medical_records=medical_records,
            criterion_list=None,
            summary=None,
            chief_complaint=None,
            node1_reasoning=None,
            documents_processed=None,
            citations_map=None,
            record_chunks=None,
            chromadb_collection=None,
            retrieved_policies=None,
            policies_by_complaint=None,
            node2_reasoning=None,
            evaluation_results=None,
            node3_reasoning=None,
            next_action="node2", 
            iteration_count=0,
            errors=[],
            user_id=user_id
        )
        
        # Run only Node 1
        state = self._node1_wrapper(initial_state)
        
        security.log_action("Agent_Process_Node1_Complete", user_id, {
            "errors": len(state.get("errors", []))
        })
        
        return state
    
    def process_node2_only(
        self,
        state: Dict,
        user_id: str
    ) -> Dict:
        """
        Process Node 2 ONLY (policy retrieval)
        Takes existing state from Node 1
        """
        security.log_action("Agent_Process_Node2_Start", user_id, {})
        
        # Run only Node 2
        state = self._node2_wrapper(state)
        
        security.log_action("Agent_Process_Node2_Complete", user_id, {
            "errors": len(state.get("errors", []))
        })
        
        return state
    
    def evaluate_criteria(
        self,
        state: Dict,
        criterion_list: List[str]
    ) -> Dict:
        """
        Evaluate criteria using existing state (with ChromaDB collection)
        Runs Node 3 only
        
        Args:
            state: Existing state from Node 1 & 2
            criterion_list: List of criteria to evaluate
            
        Returns:
            Final state after Node 3
        """
        user_id = state.get("user_id", "unknown")
        security.log_action("Agent_Evaluate_Criteria_Start", user_id, {
            "num_criteria": len(criterion_list)
        })
        
        # Update state with criteria
        state["criterion_list"] = criterion_list
        state["next_action"] = "node3"
        
        # Run Node 3 directly
        final_state = node3.process(state)
        
        security.log_action("Agent_Evaluate_Criteria_Complete", user_id, {
            "criteria_evaluated": len(criterion_list),
            "errors": len(final_state.get("errors", []))
        })
        
        return final_state


# Global agent instance
agent = ClinicalReviewAgent()
