"""
Phase 4: Travel Orchestrator with LangGraph
Stateful workflow + checkpointing + DB persistence of plans + human approval.
"""

from __future__ import annotations

import os
import uuid
import json
from typing import Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

from phases.phase4_langgraph.trip_agents import (
    TravelState,
    info_collector,
    planner,
    optimizer,
    approval,
    completion,
    error_recovery,
)

from api.datamodels import (
    Trip,
    TripRequirements,
    TravelPlan,
    OptimizationResult,
    ChatHistory,
    TripPlanModel,
)
from db import db_utils
from langgraph.checkpoint.memory import MemorySaver


# try:
#     from langgraph.checkpoint.sqlite import SqliteSaver  
#     SQLITE_CHECKPOINT_AVAILABLE = True
# except Exception:
#     SQLITE_CHECKPOINT_AVAILABLE = False


class LangGraphTripOrchestrator:
    """
    Orchestrator for Phase 4: LangGraph Stateful Workflow.

    What this orchestrator does:
    - Runs LangGraph state machine: InfoCollector -> Planner -> Optimizer -> Approval -> Completion
    - Stops when mandatory requirements are missing (ask user for missing details)
    - Persists TravelPlan into DB (trip_plans) as draft/versioned
    - Supports manual approval (pause/resume via thread_id checkpoint)
    - Updates trip status + trip plan status on approval/rejection
    """

    def __init__(self):
        load_dotenv()
        from pathlib import Path

        # if SQLITE_CHECKPOINT_AVAILABLE:
        #     ckpt_path = Path(__file__).resolve().parents[2] 
        #     self.checkpointer = SqliteSaver(str(ckpt_path))
        # else:
        self.checkpointer = MemorySaver()

        self.graph = self._build_graph()

    # -------------------------
    # Graph routing
    # -------------------------
    def _route_after_info(self, state: TravelState) -> str:
        if state.get("last_error"):
            return "error_recovery"

        req = state.get("requirements") or {}
        if req.get("mode") == "trip":
            return "planner"

        return END  

    def _route_after_planner(self, state: TravelState) -> str:
        if state.get("last_error"):
            return "error_recovery"
        return "optimizer"

    def _route_after_optimizer(self, state: TravelState) -> str:
        if state.get("last_error"):
            return "error_recovery"
        return "approval"

    def _route_after_approval(self, state: TravelState) -> str:
        if state.get("last_error"):
            return "error_recovery"

        mode = state.get("approval_mode", "auto")
        decision = state.get("approval_decision")

        # manual mode: if no decision yet -> pause here
        if mode == "manual" and decision not in ("approved", "rejected"):
            return END

        # rejected -> feedback loop to planner (new version should be created)
        if decision == "rejected":
            return "planner"

        return "completion"

    def _build_graph(self):
        g = StateGraph(TravelState)

        g.add_node("info_collector", info_collector)
        g.add_node("planner", planner)
        g.add_node("optimizer", optimizer)
        g.add_node("approval", approval)
        g.add_node("completion", completion)
        g.add_node("error_recovery", error_recovery)

        g.add_edge(START, "info_collector")

        g.add_conditional_edges("info_collector", self._route_after_info, {
            "planner": "planner",
            "error_recovery": "error_recovery",
            END: END,
        })

        g.add_conditional_edges("planner", self._route_after_planner, {
            "optimizer": "optimizer",
            "error_recovery": "error_recovery",
        })

        g.add_conditional_edges("optimizer", self._route_after_optimizer, {
            "approval": "approval",
            "error_recovery": "error_recovery",
        })

        g.add_conditional_edges("approval", self._route_after_approval, {
            "planner": "planner",
            "completion": "completion",
            "error_recovery": "error_recovery",
            END: END,
        })

        g.add_edge("completion", END)
        g.add_edge("error_recovery", END)

        return g.compile(checkpointer=self.checkpointer)

    # -------------------------
    # Trip creation
    # -------------------------
    def _create_trip_if_needed(self, state: TravelState) -> TravelState:
        """
        Create Trip record only after requirements are complete and trip_id == 0.
        Also attach pre-trip chat rows (trip_id=0) to new trip.
        """
        if state.get("trip_id", 0) and state["trip_id"] != 0:
            return state

        #req = TripRequirements(**(state.get("requirements") or {}))
        req_dict = state.get("requirements") or {}
        if req_dict.get("mode") != "trip":
            return state
        
        req = TripRequirements(**req_dict)

        user_id = state["user_id"]
        phase = state.get("phase", "phase4_langgraph")

        trip_dict = req.to_trip_dict(user_id=user_id, phase=phase, title="My Trip")
        trip = Trip(**trip_dict)

        new_id = db_utils.create_trip(trip)
        state["trip_id"] = new_id

        # attach pre-trip chats to new trip
        try:
            db_utils.attach_pre_trip_chat_to_trip(user_id=user_id, phase=phase, new_trip_id=new_id, limit=50)
        except Exception:
            pass

        try:
            db_utils.update_trip_status(new_id, "inprogress")
        except Exception:
            pass

        return state

    # -------------------------
    # Plan persistence helpers
    # -------------------------
    def _next_plan_version(self, trip_id: int) -> int:
        latest = db_utils.get_trip_plan_by_trip_id(trip_id)
        if latest and getattr(latest, "version", None):
            return int(latest.version) + 1
        return 1
    
    def _next_version_for_trip(self,trip_id: int) -> int:
        try:
            latest = db_utils.get_trip_plan_by_trip_id(trip_id, version=None)
            return (latest.version + 1) if latest and getattr(latest, "version", None) else 1
        except Exception:
            return 1

    def _persist_plan_draft(
        self,
        trip_id: int,
        travel_plan: TravelPlan,
        optimization: Optional[OptimizationResult],
        state: TravelState,
    ) -> Tuple[int, int]:
        """
        Save TravelPlan to DB (trip_plans) as a new version draft.
        Returns (plan_id, version).
        """
        version = self._next_plan_version(trip_id)
        model = TripPlanModel.from_travel_plan(travel_plan, trip_id=trip_id, version=version)

        # attach useful agent metadata (optimization + trace + tool logs)
        meta = {
                "phase": state.get("phase", "phase4_langgraph"),
                "thread_id": state.get("thread_id"),            # ? persist thread_id
                "plan_version": model.version,
                "optimization_result": optimization.model_dump() if optimization else None,
                "execution_trace": state.get("execution_trace", []),
                "tool_log": state.get("tool_log", []),
                "tool_fallback_used": state.get("tool_fallback_used", False),
            }
        model.agent_metadata = json.dumps(meta)
        model.status = "draft"

        plan_id = db_utils.create_trip_plan(model)
        return plan_id, version

    # -------------------------
    # Main: plan_trip
    # -------------------------
    def plan_trip(
        self,
        user_id: int,
        user_input: str,
        phase: str = "phase4_langgraph",
        approval_mode: str = "auto",
        previous_state: Optional[dict] = None,
    ) -> Dict[str, Any]:

        # stable thread_id across resumes
        if previous_state and previous_state.get("thread_id"):
            thread_id = previous_state["thread_id"]
        else:
            thread_id = f"phase4_{user_id}_{uuid.uuid4().hex[:10]}"

        # merge state
        state: TravelState = (previous_state or {})  # type: ignore
        state.update({
            "phase": phase,
            "user_id": user_id,
            "user_input": user_input,
            "approval_mode": approval_mode,
            "thread_id": thread_id,
            "trip_id": state.get("trip_id", 0) or 0,
        })

        # persist user chat immediately (trip_id may be 0 pre-trip)

        try:
            db_utils.save_chat_message_service(ChatHistory(
            trip_id=state.get("trip_id", 0),
            user_id=user_id,
            role="user",
            phase=phase,
            content=user_input
        ))
        except Exception as e:
            print(e)
            pass

        config = RunnableConfig(configurable={"thread_id": thread_id})

        out_state: TravelState = self.graph.invoke(state, config=config)  # type: ignore

        requirements = out_state.get("requirements") or {}
        mode = requirements.get("mode", "missing")

        if mode != "trip":
            out_state.pop("travel_plan", None)
            out_state.pop("optimization", None)
            out_state.pop("plan_id", None)
            out_state.pop("plan_version", None)
            out_state["approval_required"] = False


        # create trip if requirements complete
        try:
            out_state = self._create_trip_if_needed(out_state)
        except:
            pass



        requirements = out_state.get("requirements")
        travel_plan_dict = out_state.get("travel_plan")
        optimization_dict = out_state.get("optimization")

        print(f"printing requirment {requirements}")

        requirements = out_state.get("requirements") or {}
        mode = requirements.get("mode", "missing")

        # Stop if missing mandatory info
        if mode != "trip":
            # We still create req_obj only for returning a schema-shaped payload
            try:
                req_obj = TripRequirements(**requirements)
            except Exception:
                req_obj = TripRequirements(mode="missing")

            response: Dict[str, Any] = {
                "success": False,
                "phase": phase,
                "trip_id": out_state.get("trip_id", 0),
                "thread_id": thread_id,
                "message": out_state.get("agent_message") or "Additional information needed",
                "workflow_state": out_state,
                "execution_summary": {
                    "trace": out_state.get("execution_trace", []),
                    "tool_fallback_used": out_state.get("tool_fallback_used", False),
                    "tool_log": out_state.get("tool_log", []),
                    "last_error": out_state.get("last_error"),
                },
                "trip_requirements": req_obj.model_dump(),
                "next_action": "ask_missing_details",
                "approval_required": False,
            }

            # Prefer agent_message generated by InfoCollector (it already asks missing_fields)
            response["message"] = requirements.get("agent_message") or out_state.get("agent_message") or "Please provide missing trip details."
            return response


        if mode=='trip':

            response: Dict[str, Any] = {
                "success":True ,
                "phase": phase,
                "trip_id": out_state.get("trip_id", 0),
                "thread_id": thread_id,
                "message": out_state.get("agent_message") or "Workflow executed.",
                "workflow_state": out_state,
                "execution_summary": {
                    "trace": out_state.get("execution_trace", []),
                    "tool_fallback_used": out_state.get("tool_fallback_used", False),
                    "tool_log": out_state.get("tool_log", []),
                    "last_error": out_state.get("last_error"),
                },
                "trip_requirements": requirements,
            }
            

        

        # build TravelPlan / OptimizationResult objects (if available)
        travel_plan_obj = TravelPlan(**travel_plan_dict) if travel_plan_dict else None
        optimization_obj = OptimizationResult(**optimization_dict) if optimization_dict else None

        if travel_plan_obj:
            response["travel_plan"] = travel_plan_obj.model_dump()

        if optimization_obj:
            response["optimization_result"] = optimization_obj.model_dump()

        version=0
        plan_id=0

        # Persist plan draft once trip exists and plan is generated
        if out_state.get("trip_id") and travel_plan_obj:
            # avoid duplicating if already saved in this run
            if not out_state.get("plan_id"):

                try:
                    trip_id=out_state.get("trip_id")
                    
                    trip_details = db_utils.get_trip_with_plan(trip_id)
                    if trip_details.get('plan_version') and trip_details['plan_status']=='draft':
                        version=trip_details.get('plan_version')

                            
                    else:
                        version = self._next_version_for_trip(trip_id)
                        print(f"version:{version}")
                        print(travel_plan_obj)

                        plan_id = db_utils.save_travel_plan_to_db(travel_plan_obj, trip_id, version)
                        print(f"plan_id:{plan_id}")
                        # Keep status model-consistent: 'draft' | 'approved' | 'rejected'
                        db_utils.update_trip_plan_status(plan_id, "draft")
                except Exception as e:
                    print(e)
                    pass




                # plan_id, plan_version = self._persist_plan_draft(
                #     trip_id=out_state["trip_id"],
                #     travel_plan=travel_plan_obj,
                #     optimization=optimization_obj,
                #     state=out_state,
                # )
                out_state["plan_id"] = plan_id
                out_state["plan_version"] = version

            response["plan_id"] = out_state.get("plan_id")
            response["plan_version"] = out_state.get("plan_version")

        # Manual approval handling
        approval_required = bool(out_state.get("approval_required", False)) if approval_mode == "manual" else False
        response["approval_required"] = approval_required

        if approval_mode == "manual" and approval_required:
            response["next_action"] = "awaiting_approval"
            return response

        # Auto mode => mark completed + approve plan
        response["next_action"] = "completed"
        try:
            if out_state.get("trip_id"):
                db_utils.update_trip_status(out_state["trip_id"], "completed")
        except Exception:
            pass

        # if we have a plan saved, mark it approved in auto mode
        try:
            if out_state.get("plan_id"):
                db_utils.update_trip_plan_status(out_state["plan_id"], "approved")
        except Exception:
            pass

        return response

    # -------------------------
    # Continue after missing info
    # -------------------------
    def continue_trip_clarification(
        self,
        previous_state: dict,
        user_input: str,
        user_id: int,
        approval_mode: str = "auto",
    ) -> Dict[str, Any]:
        return self.plan_trip(
            user_id=user_id,
            user_input=user_input,
            phase=previous_state.get("phase", "phase4_langgraph"),
            approval_mode=approval_mode,
            previous_state=previous_state,
        )

    # -------------------------
    # Human approval handler
    # -------------------------
    def handle_human_approval(self, thread_id: str, decision: str, feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        decision: "approved" or "rejected"
        """
        if decision not in ("approved", "rejected"):
            return {"success": False, "message": "decision must be 'approved' or 'rejected'"}

        config = RunnableConfig(configurable={"thread_id": thread_id})

        resume_state: TravelState = {
            "thread_id": thread_id,
            "approval_decision": decision,
            "feedback": feedback,
        }

        out_state: TravelState = self.graph.invoke(resume_state, config=config)  # type: ignore

        response: Dict[str, Any] = {
            "success": True,
            "thread_id": thread_id,
            "trip_id": out_state.get("trip_id", 0),
            "decision": decision,
            "feedback": feedback,
            "workflow_state": out_state,
            "message": out_state.get("agent_message") or "Approval processed.",
            "execution_summary": {
                "trace": out_state.get("execution_trace", []),
                "tool_fallback_used": out_state.get("tool_fallback_used", False),
                "tool_log": out_state.get("tool_log", []),
                "last_error": out_state.get("last_error"),
            },
            "approval_required": bool(out_state.get("approval_required", False)),
        }

        # attach plan/optimization if present
        if out_state.get("travel_plan"):
            response["travel_plan"] = TravelPlan(**out_state["travel_plan"]).model_dump()
        if out_state.get("optimization"):
            response["optimization_result"] = OptimizationResult(**out_state["optimization"]).model_dump()

        # Update DB statuses
        try:
            trip_id = out_state.get("trip_id", 0)
            plan_id = out_state.get("plan_id")

            if trip_id:
                if decision == "approved":
                    db_utils.update_trip_status(trip_id, "completed")
                else:
                    db_utils.update_trip_status(trip_id, "in_progress")

            if plan_id:
                if decision == "approved":
                    db_utils.update_trip_plan_status(plan_id, "approved")
                else:
                    db_utils.update_trip_plan_status(plan_id, "rejected")

        except Exception:
            pass

        return response


def test_langgraph_orchestrator():
    orch = LangGraphTripOrchestrator()
    user_id = 3

    # Test single turn (complete)
    r1 = orch.plan_trip(
        user_id=user_id,
        user_input="Plan a trip to Goa from Bangalore Feb 15-18 2026 under 10000 rupess for 1 adult ",
        approval_mode="manual"
    )

    print(r1)


    print("R1 next_action:", r1.get("next_action"))
    print("R1 message:", r1.get("message"))
    print("R1 trip_id:", r1.get("trip_id"), "plan_id:", r1.get("plan_id"))

    # If awaiting approval, approve
    if r1.get("next_action") == "awaiting_approval":
        r2 = orch.handle_human_approval(r1["thread_id"], "approved", feedback="Looks good")
        print("R2 message:", r2.get("message"))
        print("R2 trip_id:", r2.get("trip_id"))

if __name__ == "__main__":
    test_langgraph_orchestrator()