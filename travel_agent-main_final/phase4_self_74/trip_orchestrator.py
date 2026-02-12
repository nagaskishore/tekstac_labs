"""
Phase 4: Travel Orchestrator with LangGraph
Complete stateful workflow implementation with checkpointing and human-in-the-loop.
"""
from __future__ import annotations
from typing import Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
import json
import time

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))
from phases.phase4_langgraph.trip_agents import (
    TravelState,
    info_collector,
    planner,
    optimizer,
    approval,
    completion,
    error_recovery
)
from db import db_utils
from api.datamodels import ChatHistory, TravelPlan


class LangGraphTripOrchestrator:
    """
    Orchestrator for Phase 4: LangGraph Stateful Workflow.
    - Uses LangGraph to define workflow states and transitions
    - Supports human-in-the-loop approval with checkpointing
    - Implements enterprise error handling and recovery
    - Ensures all outputs are validated and persisted
    """

    def __init__(self):
        self.phase = "phase4_langgraph"
        self.checkpointer = MemorySaver()  # In-memory checkpoint storage
        self.workflow = self._build_workflow()
        

    # =========================================================================
    # JSON-SAFE MESSAGE SERIALIZATION (fixes HumanMessage not JSON serializable)
    # =========================================================================

    def _serialize_messages(self, messages: Any) -> list[dict]:
        """Convert LangChain message objects into JSON-safe dicts."""
        if not messages:
            return []
        out: list[dict] = []
        for m in messages:
            # If already serialized
            if isinstance(m, dict) and "content" in m:
                out.append({"type": m.get("type", "message"), "content": m.get("content", "")})
                continue

            # LangChain messages usually have `.type` and `.content`
            m_type = getattr(m, "type", None)
            m_content = getattr(m, "content", None)

            if m_type is not None and m_content is not None:
                out.append({"type": str(m_type), "content": str(m_content)})
            else:
                # last resort
                out.append({"type": m.__class__.__name__, "content": str(m)})
        return out

    def _deserialize_messages(self, serialized: Any) -> list:
        """Convert serialized dict messages back into LangChain message objects."""
        if not serialized:
            return []

        out: list = []
        for item in serialized:
            if not isinstance(item, dict):
                # Unknown shape, skip safely
                continue
            m_type = (item.get("type") or "").lower().strip()
            content = item.get("content", "")

            # LangChain message `.type` is typically "human"/"ai"/"system"
            if m_type in ("human", "user"):
                out.append(HumanMessage(content=content))
            elif m_type in ("ai", "assistant"):
                out.append(AIMessage(content=content))
            elif m_type == "system":
                out.append(SystemMessage(content=content))
            else:
                # default to AIMessage so graph can still proceed
                out.append(AIMessage(content=content))
        return out

    def _sanitize_state_for_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a JSON-safe state that is STILL resumable:
        - We store messages as `messages_serialized` (JSON-safe)
        - We remove raw `messages` from the returned state
        """
        safe = dict(state) if isinstance(state, dict) else {}
        safe["messages_serialized"] = self._serialize_messages(safe.get("messages", []))
        safe.pop("messages", None)  # remove non-serializable objects
        return safe

    def _make_json_safe(self, obj: Any) -> Any:
        """Recursively make an object JSON serializable."""
        # Handle datetime
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Handle langchain message objects if any slipped through
        if hasattr(obj, "content") and hasattr(obj, "type"):
            return {"type": getattr(obj, "type", "message"), "content": getattr(obj, "content", "")}

        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        return obj

    # =========================================================================
    # WORKFLOW BUILD
    # =========================================================================

    def _build_workflow(self) -> Any:
        workflow = StateGraph(TravelState)

        workflow.add_node("info_collector", info_collector)
        workflow.add_node("planner", planner)
        workflow.add_node("optimizer", optimizer)
        workflow.add_node("approval", approval)
        workflow.add_node("completion", completion)
        workflow.add_node("error_recovery", error_recovery)

        workflow.set_entry_point("info_collector")

        workflow.add_conditional_edges(
            "info_collector",
            self._route_after_info_collection,
            {
                "planner": "planner",
                "end": END,
                "error": "error_recovery",
            },
        )

        workflow.add_conditional_edges(
            "planner",
            self._route_after_planning,
            {
                "optimizer": "optimizer",
                "error": "error_recovery",
            },
        )

        workflow.add_conditional_edges(
            "optimizer",
            self._route_after_optimization,
            {
                "approval": "approval",
                "error": "error_recovery",
            },
        )

        workflow.add_conditional_edges(
            "approval",
            self._route_after_approval,
            {
                "completion": "completion",
                "planner": "planner",
                "wait": END,
            },
        )

        workflow.add_edge("completion", END)
        workflow.add_edge("error_recovery", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # =========================================================================
    # ROUTING FUNCTIONS
    # =========================================================================

    def _route_after_info_collection(self, state: TravelState) -> Literal["planner", "end", "error"]:
        if state.get("workflow_status") == "failed" or state.get("error_count", 0) > 3:
            return "error"
        return "planner" if state.get("requirements_complete") else "end"

    def _route_after_planning(self, state: TravelState) -> Literal["optimizer", "error"]:
        if state.get("workflow_status") == "failed" or state.get("error_count", 0) > 3:
            return "error"
        return "optimizer" if state.get("travel_plan") else "error"

    def _route_after_optimization(self, state: TravelState) -> Literal["approval", "error"]:
        if state.get("workflow_status") == "failed" or state.get("error_count", 0) > 3:
            return "error"
        return "approval" if state.get("optimization_result") else "error"

    def _route_after_approval(self, state: TravelState) -> Literal["completion", "planner", "wait"]:
        approval_status = state.get("approval_status")
        if approval_status == "approved":
            return "completion"
        if approval_status == "rejected":
            return "planner"
        return "wait"

    # =========================================================================
    # MAIN TRIP PLANNING METHOD
    # =========================================================================

    def plan_trip(
        self,
        user_id: int,
        user_input: str,
        phase: str = "phase4_langgraph",
        approval_mode: str = "auto",
        previous_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        start_time = time.time()
        print(f"\n{'='*80}")
        print(" LANGGRAPH TRIP ORCHESTRATOR - Starting workflow")
        print(f" User ID: {user_id}")
        print(f" Input: {user_input[:100]}...")
        print(f"{'='*80}")

        try:
            thread_id = f"trip_{user_id}_{int(time.time())}"

            # Initialize or restore state
            if previous_state:
                # Copy to avoid mutating caller state
                initial_state = dict(previous_state)

                # If previous_state came from API response, messages were serialized
                if "messages" not in initial_state and "messages_serialized" in initial_state:
                    initial_state["messages"] = self._deserialize_messages(initial_state.get("messages_serialized"))

                initial_state["user_input"] = user_input
                initial_state.setdefault("conversation_history", []).append(f"user: {user_input}")
                # keep thread id if present
                thread_id = initial_state.get("thread_id", thread_id)

                # Make sure messages exists
                initial_state.setdefault("messages", []).append(HumanMessage(content=user_input))
            else:
                initial_state = {
                    "thread_id": thread_id,
                    "current_node": "start",
                    "workflow_status": "initialized",
                    "user_id": user_id,
                    "trip_id": None,
                    "user_input": user_input,
                    "conversation_history": [f"user: {user_input}"],

                    "requirements": None,
                    "requirements_complete": False,
                    "missing_fields": [],

                    "travel_plan": None,
                    "plan_version": 1,

                    "optimization_result": None,
                    "cost_analysis": {},
                    "alternatives": [],

                    "approval_required": True,
                    "approval_status": None,
                    "user_feedback": None,

                    "tool_results": {},
                    "tool_errors": [],
                    "fallback_used": [],

                    "error_count": 0,
                    "last_error": None,
                    "recovery_checkpoint": None,

                    "execution_start": datetime.now().isoformat(),
                    "execution_end": None,
                    "node_timings": {},

                    "messages": [HumanMessage(content=user_input)],
                }

            # Save user message to DB
            db_utils.save_chat_message(ChatHistory(
                trip_id=initial_state.get("trip_id"),
                user_id=user_id,
                role="user",
                phase=self.phase,
                content=user_input
            ))

            config = {"configurable": {"thread_id": thread_id}}
            print(f" Executing workflow with thread_id: {thread_id}")

            final_state = None
            for state_update in self.workflow.stream(initial_state, config):
                final_state = state_update
                if isinstance(state_update, dict):
                    for _, node_state in state_update.items():
                        if isinstance(node_state, dict):
                            current_node = node_state.get("current_node", "unknown")
                            workflow_status = node_state.get("workflow_status", "unknown")
                            print(f"    Node: {current_node} | Status: {workflow_status}")

            # Extract final state from stream result
            if isinstance(final_state, dict) and final_state:
                final_state = list(final_state.values())[0]
            else:
                final_state = initial_state

            execution_time = time.time() - start_time
            print(f"\n{'='*80}")
            print(" WORKFLOW EXECUTION COMPLETE")
            print(f" Status: {final_state.get('workflow_status', 'unknown')}")
            print(f" Trip ID: {final_state.get('trip_id', 'N/A')}")
            print(f" Execution Time: {execution_time:.2f}s")
            print(f" Errors: {final_state.get('error_count', 0)}")
            print(f" Fallbacks Used: {len(final_state.get('fallback_used', []))}")
            print(f"{'='*80}\n")

            workflow_status = final_state.get("workflow_status", "unknown")

            if workflow_status == "awaiting_approval":
                resp = self._format_approval_response(final_state, thread_id)
            elif workflow_status == "completed":
                resp = self._format_completion_response(final_state)
            elif workflow_status == "collecting_info" and not final_state.get("requirements_complete"):
                resp = self._format_clarification_response(final_state)
            elif workflow_status == "failed":
                resp = self._format_error_response(final_state)
            else:
                resp = self._format_progress_response(final_state)

            return self._make_json_safe(resp)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "message": "Workflow execution failed"}

    # =========================================================================
    # APPROVAL CONTINUATION (valid Trip status + TripPlan status)
    # =========================================================================

    def continue_trip_approval(
        self,
        trip_id: int,
        approval_decision: str,
        user_feedback: str = "",
        user_id: int = None
    ) -> Dict[str, Any]:

        try:
            decision = (approval_decision or "").strip().lower()
            if decision not in ("approved", "rejected"):
                return {"success": False, "error": "approval_decision must be 'approved' or 'rejected'"}

            if not user_id:
                trip = db_utils.get_trip_by_id(trip_id)
                if not trip:
                    return {"success": False, "error": "Trip not found"}
                user_id = trip.user_id

            plan = db_utils.get_trip_plan_by_trip_id(trip_id)
            plan_id = plan.id if plan else None

            if decision == "approved":
                db_utils.update_trip_status(trip_id, "confirmed")
                if plan_id:
                    db_utils.update_trip_plan_status(plan_id, "approved")

                db_utils.save_chat_message(ChatHistory(
                    trip_id=trip_id,
                    user_id=user_id,
                    role="user",
                    phase=self.phase,
                    content=f"Plan approved: {user_feedback}" if user_feedback else "Plan approved"
                ))

                return {
                    "success": True,
                    "trip_id": trip_id,
                    "user_id": user_id,
                    "approval": True,
                    "plan_id": plan_id,
                    "plan_status": "approved" if plan_id else None,
                    "message": "Travel plan approved successfully",
                    "workflow_status": "completed"
                }

            # rejected
            db_utils.update_trip_status(trip_id, "draft")
            if plan_id:
                db_utils.update_trip_plan_status(plan_id, "rejected")

            db_utils.save_chat_message(ChatHistory(
                trip_id=trip_id,
                user_id=user_id,
                role="user",
                phase=self.phase,
                content=f"Plan rejected: {user_feedback}" if user_feedback else "Plan rejected"
            ))

            return {
                "success": True,
                "trip_id": trip_id,
                "user_id": user_id,
                "approval": False,
                "plan_id": plan_id,
                "plan_status": "rejected" if plan_id else None,
                "message": "Travel plan rejected. Please provide feedback to revise.",
                "workflow_status": "rejected",
                "next_action": "revise_plan",
                "feedback": user_feedback
            }

        except Exception as e:
            return {"success": False, "error": str(e), "message": "Failed to process approval"}

    # =========================================================================
    # CLARIFICATION CONTINUATION
    # =========================================================================

    def continue_trip_clarification(
        self,
        previous_state: Dict[str, Any],
        user_input: str,
        user_id: int,
        approval_mode: str = "auto"
    ) -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(" CLARIFICATION CONTINUATION - Resuming workflow")
        print(f" User ID: {user_id}")
        print(f"{'='*80}")
        return self.plan_trip(
            user_id=user_id,
            user_input=user_input,
            phase=self.phase,
            approval_mode=approval_mode,
            previous_state=previous_state
        )
    # =========================================================================
    # HUMAN APPROVAL HANDLING
    # =========================================================================
    def handle_human_approval(self, thread_id: str, decision: str) -> Dict[str, Any]:
        """
        Handle human approval using thread_id (trip_{user_id}_{timestamp}).

        This is a convenience method for UI layers that only track thread_id.
        It resolves user_id from thread_id, finds the most recent active trip,
        and delegates to continue_trip_approval().
        """
        try:
            if not thread_id or "_" not in thread_id:
                return {"success": False, "error": "Invalid thread_id"}

            parts = thread_id.split("_")
            # Expected: ["trip", "<user_id>", "<timestamp>"]
            if len(parts) < 3 or parts[0] != "trip":
                return {"success": False, "error": "Invalid thread_id format"}

            try:
                user_id = int(parts[1])
            except ValueError:
                return {"success": False, "error": "Invalid user_id in thread_id"}

            decision_norm = (decision or "").strip().lower()
            if decision_norm not in ("approved", "rejected"):
                return {"success": False, "error": "Decision must be 'approved' or 'rejected'"}

            # Best available method with your current db_utils
            trip = db_utils.get_active_trip_for_user(user_id)
            if not trip:
                return {"success": False, "error": "No active trip found for this user"}

            return self.continue_trip_approval(
                trip_id=trip.id,
                approval_decision=decision_norm,
                user_feedback="",
                user_id=user_id
            )

        except Exception as e:
            return {"success": False, "error": str(e), "message": "Failed to handle approval"}



    # =========================================================================
    # RESPONSE FORMATTERS (state returned is JSON-safe and resumable)
    # =========================================================================

    def _format_approval_response(self, state: TravelState, thread_id: str) -> Dict[str, Any]:
        requirements = state.get("requirements", {}) or {}
        travel_plan = state.get("travel_plan", {}) or {}
        optimization = state.get("optimization_result", {}) or {}
        cost_analysis = state.get("cost_analysis", {}) or {}

        return {
            "success": True,
            "workflow_status": "awaiting_approval",
            "trip_id": state.get("trip_id"),
            "user_id": state.get("user_id"),
            "thread_id": thread_id,
            "checkpoint_id": state.get("recovery_checkpoint"),
            "message": "Travel plan ready for approval",
            "requires_approval": True,
            "travel_plan": {
                "destination": requirements.get("destination"),
                "origin": requirements.get("origin"),
                "dates": f"{requirements.get('trip_startdate')} to {requirements.get('trip_enddate')}",
                "travelers": f"{requirements.get('no_of_adults', 1)} adult(s), {requirements.get('no_of_children', 0)} child(ren)",
                "budget": f"{requirements.get('budget', 0)} {requirements.get('currency', 'USD')}",
                "flights": travel_plan.get("flights", []),
                "hotels": travel_plan.get("hotels", []),
                "itinerary": travel_plan.get("itinerary", ""),
                "cost_analysis": cost_analysis,
                "optimization": {
                    "recommendations": optimization.get("recommendations", []),
                    "value_adds": optimization.get("value_adds", []),
                    "cost_savings": optimization.get("cost_savings", 0),
                },
            },
            "execution_summary": {
                "nodes_executed": list(state.get("node_timings", {}).keys()),
                "execution_time": sum(state.get("node_timings", {}).values()),
                "errors": state.get("error_count", 0),
                "fallbacks_used": len(state.get("fallback_used", [])),
            },
            "state": self._sanitize_state_for_response(state),
        }

    def _format_completion_response(self, state: TravelState) -> Dict[str, Any]:
        return {
            "success": True,
            "workflow_status": "completed",
            "trip_id": state.get("trip_id"),
            "user_id": state.get("user_id"),
            "message": "Travel planning completed successfully!",
            "travel_plan": state.get("travel_plan", {}),
            "requirements": state.get("requirements", {}),
            "execution_summary": {
                "nodes_executed": list(state.get("node_timings", {}).keys()),
                "total_time": sum(state.get("node_timings", {}).values()),
                "errors": state.get("error_count", 0),
                "fallbacks_used": len(state.get("fallback_used", [])),
            },
            "state": self._sanitize_state_for_response(state),
        }

    def _format_clarification_response(self, state: TravelState) -> Dict[str, Any]:
        missing_fields = state.get("missing_fields", [])
        return {
            "success": True,
            "workflow_status": "needs_clarification",
            "trip_id": state.get("trip_id"),
            "user_id": state.get("user_id"),
            "message": "Additional information needed",
            "missing_fields": missing_fields,
            "prompt": f"Please provide the following information: {', '.join(missing_fields)}",
            "partial_requirements": state.get("requirements", {}),
            "continue_available": True,
            "state": self._sanitize_state_for_response(state),
        }

    def _format_error_response(self, state: TravelState) -> Dict[str, Any]:
        return {
            "success": False,
            "workflow_status": "failed",
            "trip_id": state.get("trip_id"),
            "user_id": state.get("user_id"),
            "error": state.get("last_error", "Unknown error"),
            "message": "Workflow encountered errors",
            "tool_errors": state.get("tool_errors", []),
            "error_count": state.get("error_count", 0),
            "recovery_available": True,
            "state": self._sanitize_state_for_response(state),
        }

    def _format_progress_response(self, state: TravelState) -> Dict[str, Any]:
        return {
            "success": True,
            "workflow_status": state.get("workflow_status", "in_progress"),
            "trip_id": state.get("trip_id"),
            "user_id": state.get("user_id"),
            "message": "Workflow in progress",
            "current_node": state.get("current_node", "unknown"),
            "progress": {
                "requirements_complete": state.get("requirements_complete", False),
                "travel_plan_created": bool(state.get("travel_plan")),
                "optimization_complete": bool(state.get("optimization_result")),
            },
            "state": self._sanitize_state_for_response(state),
        }


# =============================================================================
# TEST FUNCTION
# =============================================================================
def test_langgraph_orchestrator():
    print("\n" + "=" * 80)
    print(" TESTING LANGGRAPH ORCHESTRATOR")
    print("=" * 80)

    orchestrator = LangGraphTripOrchestrator()

    print("\n Test Case 1: Complete Query")
    result1 = orchestrator.plan_trip(
        user_id=1,
        user_input="I want to plan a leisure trip from Bangalore to Goa from February 15-18, 2026, for 2 adults with a budget of 8000 INR."
    )
    # Now result is JSON-safe
    print(f"\nResult: {json.dumps(result1, indent=2)}")

    print("\n Test Case 2: Multi-turn Conversation")
    result2a = orchestrator.plan_trip(
        user_id=2,
        user_input="I want to plan a solo business trip from Mumbai to Singapore."
    )
    print(f"\nFirst Turn Result: {json.dumps(result2a, indent=2)}")

    if result2a.get("workflow_status") == "needs_clarification":
        result2b = orchestrator.continue_trip_clarification(
            previous_state=result2a.get("state", {}),
            user_input="Next month for 4 days starting on the 15th with a budget of $1200 USD",
            user_id=2
        )
        print(f"\nSecond Turn Result: {json.dumps(result2b, indent=2)}")

    print("\n" + "=" * 80)
    print(" TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_langgraph_orchestrator()