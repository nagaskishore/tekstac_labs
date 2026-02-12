"""
FastAPI Application - LEARNING PROJECT
TODO: Complete the API implementation by connecting your AI agents

Learning Objectives:
- Learn to create FastAPI endpoints
- Understand API request/response patterns
- Integrate with agent orchestrators
- Handle different AI framework phases
"""
import sys
import os
import datetime
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from typing import Any, Optional, List, Literal, Dict, Union

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from api.datamodels import HotelSuggestion, FlightSuggestion, ApprovalRequest, TripPlanModel, TravelPlan
from api.tools import hotel_search_tool, flight_search_tool, weather_lookup_tool, datetime_tool_func, local_experience_tool
from db import db_utils
from phases.phase2_crewai.trip_orchestrator import CrewAITripOrchestrator
from phases.phase3_autogen.trip_orchestrator import AutoGenTripOrchestrator
from phases.phase4_langgraph.trip_orchestrator import LangGraphTripOrchestrator

app = FastAPI(title="TravelMate AI API", version="1.0.0")

# =============================================================================
# API ENDPOINTS
# =============================================================================

orchestrator_map={"phase2_crewai":CrewAITripOrchestrator(),"phase3_autogen":AutoGenTripOrchestrator(),"phase4_langgraph":LangGraphTripOrchestrator()}


# CORS for Streamlit
#app.add_middleware(
    #CORSMiddleware,
    #allow_origins=["*"],   
    #allow_credentials=True,
    #allow_methods=["*"],
    #allow_headers=["*"],
#)


# TODO: Implement main trip planning endpoint
@app.post("/api/v1/plan_trip")
def plan_trip(user_input: str, user_id: int,trip_id:Optional[int]=0, phase: str = "phase2_crewai",approval_mode: str = "manual", previous_state: Optional[Dict[str, Any]] = Body(default=None)):
    """
    Plan a trip using the specified AI framework
    
    Supported phases:
    - phase2_crewai: CrewAI framework with sequential agents
    - phase3_autogen: Microsoft AutoGen with group chat  
    - phase4_langgraph: LangGraph with state management
    """
    
    # TODO: Import and use orchestrators based on phase
    # TODO: Add error handling for unsupported phases
    
    phase = (phase or "").strip().lower()
    print(phase)
    _orchestrator=orchestrator_map.get(phase)
    if not _orchestrator:
        raise HTTPException(status_code=400, detail=f"Unsupported phase '{phase}'")
    # If trip_id is None -> new trip -> no history/context yet
    if trip_id ==0:
        chat_history = db_utils.get_recent_chat_history(5)
        context = {}
    else:
        # Validate trip exists (optional but recommended)
        trip = db_utils.get_trip_by_id(trip_id)
        if not trip:
            raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")

        chat_history = db_utils.load_chat_history_service(trip_id) or []
        context = db_utils.get_trip_context(user_id=user_id, trip_id=trip_id, phase=phase) or {}

    
    print(chat_history)
    print(context)

    if phase =="phase2_crewai":
        result = _orchestrator.plan_trip(user_input=user_input, user_id=user_id,trip_id=trip_id,conversation_history=chat_history,context=context)
        print(result)  
        return result
    elif phase == "phase3_autogen":
        result = _orchestrator.plan_trip(user_input=user_input, user_id=user_id,trip_id=trip_id)
        print(result)  
        return result
    elif phase == "phase4_langgraph":
        # trip_id is created inside orchestrator after mandatory requirements complete
        return _orchestrator.plan_trip(
            user_id=user_id,
            user_input=user_input,
            phase=phase,
            approval_mode=approval_mode,
            previous_state=previous_state
        )

    raise HTTPException(status_code=400, detail=f"Unsupported phase '{phase}'")


    
    

# TODO: Implement approval endpoint
@app.post("/api/v1/approve")
def approve_trip(request: ApprovalRequest):
    """Approve or reject a travel plan"""
    # TODO: Handle approval logic with your agents
    print(request)
    trip_id = request.trip_id
    phase = (request.phase or "").strip().lower()

    _orchestrator=orchestrator_map.get(phase)
    if not _orchestrator:
        raise HTTPException(status_code=400, detail=f"Unsupported phase '{phase}'")
    if trip_id is None:
        raise HTTPException(status_code=400, detail="trip_id is required")

    new_status = "approved" if request.approval else "rejected"

    print(new_status)
    print('---do this approve ===-')
    if phase != 'phase4_langgraph':
        if request.approval:
            result=_orchestrator.continue_trip_approval(trip_id=request.trip_id,user_id=request.user_id,approval_decision=new_status, user_feedback=request.feedback)
            return result
        else:
            chat_history = db_utils.load_chat_history_service(trip_id) or []
            context = db_utils.get_trip_context(user_id=request.user_id, trip_id=request.trip_id, phase=phase) or {}
            #result=_orchestrator.plan_trip(user_input=request.feedback, user_id=request.user_id,trip_id=request.trip_id,conversation_history=chat_history,context=context,approval_callback=True)
            result=_orchestrator.continue_trip_approval(trip_id=request.trip_id,user_id=request.user_id,approval_decision=new_status, user_feedback=request.feedback)

            return result
    
    if phase == "phase4_langgraph":
        thread_id = request.thread_id

        # ? If UI didn?t send thread_id, try DB fallback
        if not thread_id and request.trip_id:
            thread_id = db_utils.get_latest_thread_id_for_trip(request.trip_id)

        if not thread_id:
            raise HTTPException(status_code=400, detail="thread_id is required for phase4 approval (not found in DB)")

        decision = "approved" if request.approval else "rejected"
        return _orchestrator.handle_human_approval(
            thread_id=thread_id,
            decision=decision,
            feedback=request.feedback,
        )


# TODO: Implement health check endpoint
@app.get("/")
@app.get("/api/v1/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "TravelMate AI API"}

# Trip plan management endpoints
@app.get("/api/v1/trip/{trip_id}/plan")
def get_trip_plan(trip_id: int, version: Optional[int] = None):
    """Get trip plan by trip ID"""
    try:
        trip_plan = db_utils.get_trip_plan_by_trip_id(trip_id, version)
        if trip_plan:
            return {
                "success": True,
                "plan": trip_plan.to_travel_plan().dict(),
                "metadata": {
                    "trip_id": trip_plan.trip_id,
                    "version": trip_plan.version,
                    "status": trip_plan.status,
                    "generated_at": trip_plan.generated_at
                }
            }
        else:
            return {"success": False, "error": "Trip plan not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/v1/trip/{trip_id}/plan")
def save_trip_plan(trip_id: int, travel_plan: TravelPlan, version: int = 1):
    """Save a trip plan"""
    try:
        plan_id = db_utils.save_travel_plan_to_db(travel_plan, trip_id, version)
        return {
            "success": True, 
            "plan_id": plan_id,
            "message": f"Trip plan saved for trip {trip_id}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.put("/api/v1/trip-plan/{plan_id}/status")
def update_plan_status(plan_id: int, status: str):
    """Update trip plan status"""
    try:
        updated = db_utils.update_trip_plan_status(plan_id, status)
        if updated:
            return {"success": True, "message": f"Plan status updated to {status}"}
        else:
            return {"success": False, "error": "Plan not found or update failed"}
    except Exception as e:
        return {"success": False, "error": str(e)}



if __name__ == "__main__":
    print("TravelMate AI API - Learning Project")
    print("[LEARNING PROJECT] Complete the FastAPI implementation by connecting your AI agents")
    print("Hint: Use 'uvicorn api.app:app --reload' to run the API server")