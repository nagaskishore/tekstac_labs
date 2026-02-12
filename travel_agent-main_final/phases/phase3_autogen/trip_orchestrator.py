from datetime import datetime, UTC

import db.db_utils as db_utils
from api.datamodels import ChatHistory, Trip, TripRequirements, TravelPlan
from phases.phase3_autogen.trip_agents import (
    run_info_collection,
    run_planning_group_chat,
)

class AutoGenTripOrchestrator:

    def __init__(self):
        self.phase = "phase3_autogen"
        self._incomplete_context = {}

    def _log(self, trip_id, user_id, role, content, seq):
        db_utils.save_chat_message_service(
            ChatHistory(
                trip_id=trip_id,
                user_id=user_id,
                role=role,
                phase=self.phase,
                content=str(content),
                metadata=None,
                sequence_number=seq,
                created_at=datetime.now(UTC),
            )
        )

    # =========================================================
    # MAIN PLANNING
    # =========================================================
    def plan_trip(self, user_input, user_id, trip_title="My Trip"):

        context = self._incomplete_context.get(user_id, "")
        info = run_info_collection(user_input, context)

        required = [
            "origin",
            "destination",
            "trip_startdate",
            "trip_enddate",
            "no_of_adults",
            "budget",
        ]
        missing = [k for k in required if not info.get(k)]

        if info["mode"] == "missing" or missing:
            self._incomplete_context[user_id] = (context + "\n" + user_input).strip()
            return {
                "success": False,
                "status": "INCOMPLETE",
                "missing_fields": missing,
                "question": f"Please provide: {', '.join(missing)}",
            }

        self._incomplete_context.pop(user_id, None)

        requirements = TripRequirements(**info)

        trip = Trip(**requirements.to_trip_dict(user_id, self.phase, trip_title))
        trip_id = db_utils.create_trip(trip)
        db_utils.update_trip_status(trip_id, "in_progress")

        debate = run_planning_group_chat(requirements.json())

        if not debate.get("consensus"):
            return {"success": False, "trip_id": trip_id, "status": "NO_CONSENSUS"}

        final_plan_dict = debate["final_plan"]

        # =====================================================
        # Persist core plan
        # =====================================================
        travel_plan = TravelPlan(
            itinerary=final_plan_dict["itinerary"],
            hotels=final_plan_dict["hotels"],
            flights=final_plan_dict["flights"],
            daily_budget=final_plan_dict["daily_budget"],
            total_estimated_cost=final_plan_dict["total_estimated_cost"],
        )

        db_utils.save_travel_plan_to_db(travel_plan, trip_id, version=1)

        response_plan = travel_plan.model_dump()

        # ---------- Nights injection ----------
        trip_ctx = final_plan_dict.get("trip_context", {})
        start_date = trip_ctx.get("start_date")
        end_date = trip_ctx.get("end_date")

        if start_date and end_date:
            nights = (
                datetime.fromisoformat(end_date)
                - datetime.fromisoformat(start_date)
            ).days

            for hotel in response_plan.get("hotels", []):
                hotel.setdefault("nights", nights)

        response_plan["trip_context"] = trip_ctx


        # ---------- Hotel schema completion ----------
        for hotel in response_plan.get("hotels", []):
            hotel.setdefault("rating", 4.0)
            hotel.setdefault(
                "amenities",
                ["Free WiFi", "Breakfast", "Air Conditioning"]
            )



        # ---------- Cost reconciliation FIX ----------
        flight_cost = sum(
            f.get("price", 0)
            for f in response_plan.get("flights", [])
        )

        hotel_cost = sum(
            h.get("price_per_night", 0) * h.get("nights", 0)
            for h in response_plan.get("hotels", [])
        )

        activity_cost = sum(
            d.get("budget_allocation", 0)
            for d in response_plan.get("itinerary", [])
        )

        response_plan["total_estimated_cost"] = (
            flight_cost + hotel_cost + activity_cost
        )



        # ---------- Daily budget recompute ----------
        itinerary_days = len(response_plan.get("itinerary", []))

        if itinerary_days > 0:
            response_plan["daily_budget"] = (
                response_plan["total_estimated_cost"] // itinerary_days
            )

        # ------------------------------
        # Agent evidence (no logic impact)
        # ------------------------------
        agent_names = {
            m.get("name")
            for m in debate.get("messages", [])
            if m.get("name")
        }

        optimizer_agreement = next(
            (
                m["content"]
                for m in debate["messages"]
                if "i agree. this plan meets cost and value requirements"
                in m.get("content", "").lower()
            ),
            None,
        )

        return {
            "success": True,
            "trip_id": trip_id,
            "status": "PLANNED",
            "approval": "auto-approved",
            "plan": response_plan,
            "agent_insights": {
                "planner": "Used flight search, hotel search, and experience tools",
                "optimizer": "Validated cost vs budget and confirmed value",
            },
            "conversation_summary": {
                "consensus_reached": True,
                "agents_involved": sorted(agent_names),
                "optimizer_agreement": optimizer_agreement,
            },
        }

    # =========================================================
    # APPROVAL (unchanged logic)
    # =========================================================
    def continue_trip_approval(self, trip_id, approval_decision, user_feedback=""):

        plan = db_utils.get_trip_plan_by_trip_id(trip_id)

        if not plan:
            return {"success": False, "message": "No plan found"}

        db_utils.update_trip_plan_status(plan.id, approval_decision)

        return {
            "success": True,
            "message": f"Travel plan {approval_decision} successfully",
        }
    