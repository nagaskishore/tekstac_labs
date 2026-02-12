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
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from typing import Optional, Dict, Any
from phases.phase2_crewai.trip_orchestrator import CrewAITripOrchestrator
from phases.phase3_autogen.trip_orchestrator import AutoGenTripOrchestrator
from phases.phase4_langgraph.trip_orchestrator import LangGraphTripOrchestrator


PHASE_ORCHESTRATOR_MAP = {
    "phase2_crewai": CrewAITripOrchestrator(),
    "phase3_autogen": AutoGenTripOrchestrator(),
    "phase4_langgraph": LangGraphTripOrchestrator(),
}


from fastapi import FastAPI
from typing import Optional, List
from api.datamodels import HotelSuggestion, FlightSuggestion, ApprovalRequest, TripPlanModel, TravelPlan
from api.tools import hotel_search_tool, flight_search_tool, weather_lookup_tool, datetime_tool_func, local_experience_tool
from db import db_utils



app = FastAPI(title="TravelMate AI API", version="1.0.0")



# =============================================================================
# API ENDPOINTS
# =============================================================================

from pydantic import BaseModel

class PlanTripRequest(BaseModel):
    user_input: str
    user_id: int
    phase: str = "phase4_langgraph"
    previous_state: Optional[Dict[str, Any]] = None


# TODO: Implement main trip planning endpoint
@app.post("/api/v1/plan_trip")
def plan_trip(request: PlanTripRequest):

    """
    Plan a trip using the specified AI framework
    
    Supported phases:
    - phase2_crewai: CrewAI framework with sequential agents
    - phase3_autogen: Microsoft AutoGen with group chat  
    - phase4_langgraph: LangGraph with state management
    """
    
    # TODO: Import and use orchestrators based on phase
    # TODO: Add error handling for unsupported phases
    if request.phase not in PHASE_ORCHESTRATOR_MAP:

        return {
            "success": False,
            "error": f"Unsupported phase: {request.phase}"
        }

    try:
        orchestrator = PHASE_ORCHESTRATOR_MAP[request.phase]


        result = orchestrator.plan_trip(
            user_input=request.user_input,
            user_id=request.user_id,
            previous_state=request.previous_state
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

    
    #return "[LEARNING PROJECT] Implement your AI agent orchestrators here" # Placeholder return

# TODO: Implement approval endpoint
@app.post("/api/v1/approve")
def approve_trip(request: ApprovalRequest):
    """Approve or reject a travel plan"""
    # TODO: Handle approval logic with your agents
    if request.phase not in PHASE_ORCHESTRATOR_MAP:
        return {"success": False, "error": "Unsupported phase"}

    try:
        orchestrator = PHASE_ORCHESTRATOR_MAP[request.phase]

        decision = "approved" if request.approval else "rejected"

        result = orchestrator.continue_trip_approval(
            trip_id=request.trip_id,
            approval_decision=decision,
            user_feedback=request.feedback or "",
            user_id=request.user_id,
        )

        # fetch latest saved plan for response metadata
        plan = db_utils.get_trip_plan_by_trip_id(request.trip_id)

        return {
            "success": True,
            "trip_id": request.trip_id,
            "user_id": request.user_id,
            "approval": request.approval,
            "feedback": request.feedback,
            "updated_status": decision,
            "plan_id": plan.id if plan else None,
            "message": result.get("message", "Approval processed"),
            "plan_status": result.get("plan_status"),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
    

    #return "[LEARNING PROJECT] Implement approval handling logic here" # Placeholder return

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
                "plan": trip_plan.to_travel_plan().model_dump(),
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
    """
    Update trip plan status.
    Allowed: draft | confirmed | approved | rejected
    """
    try:
        status = status.lower().strip()

        if status not in {"draft", "confirmed", "approved", "rejected"}:
            return {
                "success": False,
                "error": "Invalid status value"
            }

        updated = db_utils.update_trip_plan_status(
            plan_id=int(plan_id),
            status=status
        )

        if not updated:
            return {
                "success": False,
                "error": "Plan not found or update failed"
            }

        return {
            "success": True,
            "plan_id": plan_id,
            "updated_status": status
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }




if __name__ == "__main__":
    print("TravelMate AI API - Learning Project")
    print("[LEARNING PROJECT] Complete the FastAPI implementation by connecting your AI agents")
    print("Hint: Use 'uvicorn api.app:app --reload' to run the API server")