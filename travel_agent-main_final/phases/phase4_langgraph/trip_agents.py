"""
Phase 4: Travel Agents with LangGraph
LLM-driven nodes with intelligent tool-calling + structured outputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict, Annotated, Any
import os
import json

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from api.datamodels import TripRequirements, TravelPlan, OptimizationResult, ChatHistory
from db import db_utils

from toolkits.web_search_service import WebSearchService
from toolkits.weather_tool import WeatherTool
from toolkits.amadeus_hotel_search import AmadeusHotelToolkit
from toolkits.amadeus_flight_tool import AmadeusFlightToolkit
from toolkits.amadeus_experience_tool import AmadeusExperienceToolkit
from toolkits.current_datetime import DateTimeTool


# -----------------------------
# Shared LangGraph State
# -----------------------------
class TravelState(TypedDict, total=False):
    # Routing / identity
    phase: str
    user_id: int
    trip_id: int
    thread_id: str

    # Conversation
    user_input: str
    messages: Annotated[List[Any], add_messages]

    # Workflow artifacts 
    requirements: Dict[str, Any]
    travel_plan: Dict[str, Any]
    optimization: Dict[str, Any]

    # Control flags
    requirements_complete: bool
    approval_mode: str  # "auto" | "manual"
    approval_required: bool
    approval_decision: Optional[str]  # "approved" | "rejected" | None
    feedback: Optional[str]

    # Observability
    last_error: Optional[str]
    tool_fallback_used: bool
    tool_log: List[Dict[str, Any]]
    execution_trace: List[str]

    # UI-facing message
    agent_message: Optional[str]


# -----------------------------
# TravelAgents
# -----------------------------
class TravelAgents:
    """
    LangGraph nodes:
    - info_collector_node -> TripRequirements (structured)
    - planner_node        -> TravelPlan (structured, tool-grounded)
    - optimizer_node      -> OptimizationResult (structured, tool-grounded)
    - approval_node       -> pauses for manual approval
    - completion_node     -> marks completion
    - error_recovery_node -> error handler
    """

    def __init__(self):
        load_dotenv()

        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.2,
        )

        # Tool backends
        self.web_search = WebSearchService()
        self.weather = WeatherTool()
        self.hotel_toolkit = AmadeusHotelToolkit()
        self.flight_toolkit = AmadeusFlightToolkit()
        self.experience_toolkit = AmadeusExperienceToolkit()
        self.datetime_tool = DateTimeTool()

  



        def get_current_datetime(self) -> Dict[str, Any]:
            """Get current datetime as JSON. Returns {'date': 'YYYY-MM-DD', ...}."""
            return self.datetime_tool.get_current_datetime()


        def web_search(self,query: str, max_results: int = 5) -> Dict[str, Any]:
            """Web search fallback. Args: query, max_results. Returns {query, results:[...]}."""
            return self.web_search.search(query=query, max_results=max_results)


        def get_weather_range(self,city: str, start_date: str, end_date: str, country_hint: str = "India") -> Dict[str, Any]:
            """
            Weather by city/date-range. Bias with country to avoid Goa->Genoa.
            Args: city, start_date(YYYY-MM-DD), end_date(YYYY-MM-DD), country_hint(default India)
            """
            place = city.strip()
            if country_hint and "," not in place:
                place = f"{place}, {country_hint}"
            return self.weather_tool.get_weather_range(place=place, start_date=start_date, end_date=end_date)


        def flight_search(self,origin_city: str, dest_city: str, departure_date: str, return_date: Optional[str] = None, adults: int = 1) -> List[Dict[str, Any]]:
            """Search flights. Args: origin_city, dest_city, departure_date, return_date(optional), adults."""
            return self.flight_toolkit.flight_search(
                origin_city=origin_city,
                dest_city=dest_city,
                departure_date=departure_date,
                return_date=return_date,
                adults=adults
            )


        


        def hotel_search(self,city:str, checkin: str, checkout: str, adults: int = 1) -> List[Dict[str, Any]]:
            """
            Search hotel offers.
            Args: hotel_ids, hotels, checkin, checkout, adults
            """
            # IMPORTANT: make sure your hotel_search implementation does NOT iterate on None
            hotel_ids, hotels = self.hotel_toolkit.hotel_list(city_name=city, radius=10)
            return self.hotel_toolkit.hotel_search(
                hotel_ids=hotel_ids[:3],
                hotels=hotels[:3],
                check_in_date=checkin,
                check_out_date=checkout,
                adults=adults
            )


        def experience_search(self, city: str, radius_km: int = 20, max_results: int = 10) -> List[Dict[str, Any]]:
            """Search experiences. Args: city, radius_km, max_results."""
            return self.experience_toolkit.experience_search(city_name=city, radius_km=radius_km, max_results=max_results)

        # ---- LangChain tool wrappers ----
        self.t_get_current_datetime = StructuredTool.from_function(
            func=get_current_datetime,
            name="get_current_datetime",
            description="Get current datetime as JSON. Returns {'date': 'YYYY-MM-DD', ...}. Use to resolve relative dates.",
        )

        self.t_web_search = StructuredTool.from_function(
            func=web_search,
            name="web_search",
            description="Web search fallback. Args: query:str, max_results:int. Returns {query, results:[...]}",
        )

        self.t_weather_range = StructuredTool.from_function(
            func=get_weather_range,
            name="get_weather_range",
            description="Get weather for a city over a date range. Args: city:str, start_date:str, end_date:str.",
        )

        self.t_flight_search = StructuredTool.from_function(
            func=flight_search,
            name="flight_search",
            description="Search flights. Args: origin:str, destination:str, departure_date:str, return_date:Optional[str], adults:int",
        )

        

        self.t_hotel_search = StructuredTool.from_function(
            func=hotel_search,
            name="hotel_search",
            description="Search hotel offers. Args: city: str, checkin:str, checkout:str, adults:int",
        )

        self.t_experience_search = StructuredTool.from_function(
            func=experience_search,
            name="experience_search",
            description="Search experiences. Args: city:str, radius_km:int, max_results:int",
        )

        # ---- Tool-enabled LLMs (tool step) ----
        self.llm_info_tools = self.llm.bind_tools([self.t_get_current_datetime])

        self.llm_planner_tools = self.llm.bind_tools([
            self.t_flight_search,
            
            self.t_hotel_search,
            self.t_weather_range,
            self.t_experience_search,
            self.t_get_current_datetime,
            self.t_web_search,
        ])

        self.llm_optimizer_tools = self.llm.bind_tools([self.t_web_search])

        # ---- Structured-output LLMs  ----
        
        self.llm_req_struct = self.llm.with_structured_output(TripRequirements)
        self.llm_plan_struct = self.llm.with_structured_output(TravelPlan)
        self.llm_opt_struct = self.llm.with_structured_output(OptimizationResult)

    
    def _append_trace(self, state: TravelState, text: str):
        state.setdefault("execution_trace", [])
        state["execution_trace"].append(text)

    def _tool_log(self, state: TravelState, name: str, ok: bool, detail: Any):
        state.setdefault("tool_log", [])
        state["tool_log"].append({"tool": name, "ok": ok, "detail": detail})

    def _log_chat(
        self,
        trip_id: int,
        user_id: int,
        phase: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ):
        try:
            msg = ChatHistory(
                trip_id=trip_id,
                user_id=user_id,
                role=role,
                phase=phase,
                content=content,
                metadata=json.dumps(metadata) if metadata else None,
            )
            db_utils.save_chat_message_service(msg)
        except Exception:
            pass

    def _history_as_text(self, trip_id: int, user_id: int, phase: str, limit: int = 2) -> str:
        """
        - trip_id != 0: read trip history
        - trip_id == 0: read last N user messages from pre-trip buffer
        """
        try:
            if trip_id:
                msgs = db_utils.load_chat_history_service(trip_id) or []
            else:
                msgs = db_utils.get_last_user_messages_pre_trip(user_id=user_id, phase=phase, limit=limit) or []

            lines = []
            for m in msgs:
                content = m.get("content")
                if content:
                    lines.append(f"{m.get('role')}: {content}")
            return "\n".join(lines[-limit:])
        except Exception:
            return ""

    def _get_context(self, user_id: int, trip_id: int, phase: str) -> Dict[str, Any]:
        try:
            if trip_id:
                return db_utils.get_trip_context(user_id=user_id, trip_id=trip_id, phase=phase) or {}
        except Exception:
            pass
        return {}

    # -------------------------
    # Tool-call loop (minimal + version-safe)
    # -------------------------
    def _run_tool_call_loop(
        self,
        llm_with_tools,
        messages: List[Any],
        allowed_tools: Dict[str, Any],
        state: TravelState,
        max_steps: int = 3,
        force_final_prompt: str = "Reply READY. Do not call tools.",
    ):
        """
        Executes tool calls until the model stops calling tools or max_steps is reached.
        """
        cur = list(messages)
        last = None

        for _ in range(max_steps):
            last = llm_with_tools.invoke(cur)
            tool_calls = getattr(last, "tool_calls", None) or []

            
            if not tool_calls:
                return last

            tool_messages: List[ToolMessage] = []
            for tc in tool_calls:
                name = tc.get("name")
                if name == "web_search":
                    state["tool_fallback_used"] = True
                tool_call_id = tc.get("id")
                args = tc.get("args") or {}

                fn = allowed_tools.get(name)
                if not fn:
                    result = {"error": f"Tool '{name}' not allowed here"}
                    self._tool_log(state, f"TOOL_BLOCKED:{name}", False, result)
                else:
                    try:
                        result = fn(**args) if isinstance(args, dict) else fn(args)
                        self._tool_log(state, name, True, {"args": args, "preview": str(result)[:600]})
                    except Exception as e:
                        result = {"error": str(e)}
                        self._tool_log(state, name, False, {"args": args, "error": str(e)})

                tool_messages.append(
                    ToolMessage(tool_call_id=tool_call_id, content=json.dumps(result, default=str))
                )

            cur = cur + [last] + tool_messages

        
        cur = cur + [SystemMessage(content=force_final_prompt)]
        return self.llm.invoke(cur)

    # ==========================================================
    # NODE 1: InfoCollector (Structured TripRequirements)
    # ==========================================================
    def info_collector_node(self, state: TravelState) -> TravelState:
        phase = state.get("phase", "phase4_langgraph")
        user_id = int(state.get("user_id", 0))
        trip_id = int(state.get("trip_id", 0))
        user_text = state.get("user_input") or ""

        self._append_trace(state, "info_collector_node: start")
        self._log_chat(trip_id, user_id, phase, "user", user_text, {"node": "info_collector"})

        # Reset downstream artifacts on new user turn
        state.pop("travel_plan", None)
        state.pop("optimization", None)
        state.pop("plan_id", None)
        state.pop("plan_version", None)
        state["approval_required"] = False
        state["approval_decision"] = None
        state["tool_fallback_used"] = False
        state.setdefault("tool_log", [])
        state.setdefault("execution_trace", [])

        try:
            history = self._history_as_text(trip_id=trip_id, user_id=user_id, phase=phase, limit=2)
            context = self._get_context(user_id=user_id, trip_id=trip_id, phase=phase) if trip_id else {}

            
            tool_prompt = f"""
ROLE: Travel Requirements Specialist (InfoCollector)

Tool available: get_current_datetime

INPUTS:
USER_MESSAGE:
{user_text}

CONVERSATION_HISTORY:
{history}

CONTEXT:
{json.dumps(context, default=str)}

MANDATORY DATE RULES:
- If the user mentions ANY relative date phrase such as:
  "next month", "next week", "this weekend", "starting on the 15th", "for 4 days"
  you MUST call get_current_datetime EXACTLY ONCE.
- You will use the returned 'date' as today's reference in the final extraction step.

Do NOT output TripRequirements in this step.
Finish by replying READY.
"""
            allowed = {"get_current_datetime": self.datetime_tool.get_current_datetime}

            _ = self._run_tool_call_loop(
                llm_with_tools=self.llm_info_tools,
                messages=[SystemMessage(content=tool_prompt)],
                allowed_tools=allowed,
                state=state,
                max_steps=2,
                force_final_prompt="Reply READY. Do not call tools.",
            )

            # ----- FINALIZE STEP (Structured output) -----
            finalize_prompt = f"""
You MUST return a TripRequirements object (structured).

INPUTS:
USER_MESSAGE:
{user_text}

CONVERSATION_HISTORY:
{history}

CONTEXT:
{json.dumps(context, default=str)}

TOOL_LOG_EVIDENCE:
{json.dumps(state.get("tool_log", []), default=str)[:12000]}

MANDATORY fields for mode="trip":
- origin
- destination
- trip_startdate (YYYY-MM-DD)
- trip_enddate (YYYY-MM-DD)
- budget

RELATIVE DATE RESOLUTION (NON-NEGOTIABLE):
Use get_current_datetime() output in TOOL_LOG_EVIDENCE to resolve:
- "next month starting on the 15th for 4 days" =>
  start_date = 15th day of next month relative to today's date
  end_date = start_date + 3 days
- "next month for 4 days starting on the 15th" follows same logic.
- If user gives "for N days" and a start day/date, end_date = start + (N-1) days.

CURRENCY/BUDGET RULES:
- Parse "$1200 USD" -> budget=1200, currency="USD"
- Parse "8000 INR"/"?8000"/"Rs 8000"/"rupees" -> budget=8000, currency="INR"

TRAVELERS RULE:
- "for 2 adults" / "2 travelers" -> no_of_adults=2

RULES:
- Do NOT guess missing values.
- If any mandatory field is missing -> mode="missing"
  - error="MISSING"
  - missing_fields lists ONLY missing mandatory keys
  - agent_message asks user for those keys
- If all mandatory fields present -> mode="trip", error=null, missing_fields=null, agent_message=null
"""

            req: TripRequirements = self.llm_req_struct.invoke([SystemMessage(content=finalize_prompt)])

            # Persist into state
            state["requirements"] = req.model_dump(exclude_none=False, exclude_unset=False, exclude_defaults=False)
            state["requirements_complete"] = (req.mode == "trip")
            state["agent_message"] = req.agent_message or ("Trip requirements captured successfully." if req.mode == "trip" else "Additional information needed.")

            self._append_trace(state, f"info_collector_node: mode={req.mode}")
            self._log_chat(trip_id, user_id, phase, "assistant", state["agent_message"], {"node": "info_collector", "requirements": state["requirements"]})
            return state

        except Exception as e:
            state["last_error"] = f"InfoCollector error: {e}"
            self._append_trace(state, f"info_collector_node: exception={e}")
            state["agent_message"] = "I hit an error while extracting trip requirements. Please try again."
            return state

    # ==========================================================
    # NODE 2: Planner (Tool-grounded + Structured TravelPlan)
    # ==========================================================
    def planner_node(self, state: TravelState) -> TravelState:
        self._append_trace(state, "planner_node: start")

        req_dict = state.get("requirements") or {}
        if req_dict.get("mode") != "trip":
            state["agent_message"] = req_dict.get("agent_message") or "Missing mandatory trip details."
            state.pop("travel_plan", None)
            state.pop("optimization", None)
            state["approval_required"] = False
            self._append_trace(state, "planner_node: skipped (mode != trip)")
            return state

        req = TripRequirements(**req_dict)
        state["tool_log"] = []
        destination=req.destination


        try:
            # ----- TOOL STEP (gather evidence) -----
            gather_prompt = f"""
ROLE: Planner Agent (Tool-Grounded)

YOU MUST CALL ALL TOOLS at least once (in this order is recommended):
1) flight_search(origin, destination, departure_date, return_date, adults)
2)  hotel_search(city, checkin, checkout, adults)
3) get_weather_range(city, start_date, end_date)
4) experience_search(city, radius_km, max_results)

If any tool returns empty/error:
- call web_search to fetch fallback options (e.g., "budget flights {req.origin} from {req.destination} on {req.trip_startdate}", "best budget hotels in Goa").
- If experience_search returns empty, you MUST call web_search for "top activities in {req.destination} " and use that as fallback.

TRIP REQUIREMENTS (authoritative):
{json.dumps(req.model_dump(), default=str)}. Consider the trip requirements while planning the day to day itinenaries 

IMPORTANT:
- Use adults=req.no_of_adults (default 1 if missing).
- Use departure_date=req.trip_startdate and return_date=req.trip_enddate when applicable.
- Do NOT output TravelPlan in this tool step.
- After all tool calls are done, reply with the single word: READY
"""
            allowed = {
                "flight_search": self.flight_toolkit.flight_search,
                "hotel_list": self.hotel_toolkit.hotel_list,
                "hotel_search": self.hotel_toolkit.hotel_search,
                "get_weather_range": self.weather.get_weather_range,
                "experience_search": self.experience_toolkit.experience_search,
                "get_current_datetime": self.datetime_tool.get_current_datetime,
                "web_search": self.web_search.search,
            }

            _ = self._run_tool_call_loop(
                llm_with_tools=self.llm_planner_tools,
                messages=[SystemMessage(content=gather_prompt)],
                allowed_tools=allowed,
                state=state,
                max_steps=6,  
                force_final_prompt="Reply READY. Do not call tools.",
            )

            # ----- FINALIZE STEP (Structured TravelPlan) -----
            finalize_prompt = f"""
You MUST return a TravelPlan object (structured). NO extra text.

TravelPlan schema:
- itinerary: MUST be a LIST of DICTS (not a plain string) to include activities and costs
- hotels: List[HotelSuggestion] (NON-EMPTY)
- flights: List[FlightSuggestion] (NON-EMPTY)
- daily_budget: float
- total_estimated_cost: float (should be set)

TRIP REQUIREMENTS:
{json.dumps(req.model_dump(), default=str)}

TOOL_LOG_EVIDENCE:
{json.dumps(state.get("tool_log", []), default=str)[:12000]}

STRICT OUTPUT REQUIREMENTS (RUBRIC):
1) flights must contain at least 2 FlightSuggestion items (if possible).
2) hotels must contain at least 2 HotelSuggestion items (if possible).
3) itinerary MUST be a list of dicts with keys like:
   - day (int)
   - date (YYYY-MM-DD)
   - title (string)
   - activities (list of dicts with {{"name":..., "time":..., "cost_estimate":...}})
   - weather (short string)
   - costs (dict: transport/meals/activities)
4) daily_budget:
   - if budget exists: budget / number_of_days
5) total_estimated_cost:
   - sum of: selected flight + hotel_nights*price_per_night + meals + transport + activities
   - keep within budget if possible, otherwise explain tradeoffs in itinerary notes

HOW TO USE TOOL RESULTS:
- If flight_search returns empty, use web_search results to infer at least 2 flight options:
  - airline: Airline name from websearch result
  - price: use a reasonable estimate within budget (do NOT leave 0)
  - duration: "5h 30m" ok if unknown
  - stops: 0 or 1 estimate
- If hotel_search returns empty, use web_search results to build at least 2 hotel options:
  - name from result title if available
  - price_per_night estimate (do NOT leave 0)
  - location: "{req.destination}"
- If experience_search returns empty, use web_search to add 4-6 popular activities as itinerary activities with small cost estimates.

IMPORTANT:
- Do NOT output price=0 unless absolutely unavoidable.
- Keep flights/hotels NON-EMPTY always.
"""


            plan: TravelPlan = self.llm_plan_struct.invoke([SystemMessage(content=finalize_prompt)])

            state["travel_plan"] = plan.model_dump(exclude_none=False, exclude_unset=False, exclude_defaults=False)
            state["agent_message"] = "Travel plan generated successfully."
            self._append_trace(state, "planner_node: complete")
            return state

        except Exception as e:
            state["last_error"] = f"Planner error: {e}"
            self._append_trace(state, f"planner_node: exception={e}")
            state["agent_message"] = "I hit an error while generating the travel plan. Please retry."
            return state

    # ==========================================================
    # NODE 3: Optimizer (Tool-grounded + Structured OptimizationResult)
    # ==========================================================
    def optimizer_node(self, state: TravelState) -> TravelState:
        self._append_trace(state, "optimizer_node: start")

        if not state.get("travel_plan"):
            state["agent_message"] = "Cannot optimize: travel plan not available yet."
            state["approval_required"] = False
            self._append_trace(state, "optimizer_node: skipped (no plan)")
            return state

        req = TripRequirements(**(state.get("requirements") or {}))
        plan = TravelPlan(**(state.get("travel_plan") or {}))

        state.setdefault("tool_log", [])
        state["tool_fallback_used"] = False

        try:
            
            tool_prompt = f"""
ROLE: Optimizer Agent (Tool-Grounded)

You MUST call web_search at least once to find budget-saving ideas for:
- route: {req.origin} -> {req.destination}
- dates: {req.trip_startdate} to {req.trip_enddate}
- budget: {req.budget} {getattr(req, "currency", "USD")}

Do NOT output final OptimizationResult in this step.
Finish by replying READY.
"""
            allowed = {"web_search": self.web_search.search}

            _ = self._run_tool_call_loop(
                llm_with_tools=self.llm_optimizer_tools,
                messages=[SystemMessage(content=tool_prompt)],
                allowed_tools=allowed,
                state=state,
                max_steps=3,
                force_final_prompt="Reply READY. Do not call tools.",
            )

            # ----- FINALIZE STEP (Structured OptimizationResult) -----
            finalize_prompt = f"""
You MUST return an OptimizationResult object (structured).

REQUIREMENTS:
{json.dumps(req.model_dump(), default=str)}

CURRENT PLAN:
{json.dumps(plan.model_dump(), default=str)[:12000]}

TOOL_LOG_EVIDENCE:
{json.dumps(state.get("tool_log", []), default=str)[:12000]}

STRICT RULES:
- recommendations: list[str]
- value_adds: list[str]
- cost_savings: float
- final_plan: string summary
- approval_required: MUST be true (manual approval integration)
"""

            opt: OptimizationResult = self.llm_opt_struct.invoke([SystemMessage(content=finalize_prompt)])

            # Enforce approval_required 
            opt.approval_required = True

            state["optimization"] = opt.model_dump(exclude_none=False, exclude_unset=False, exclude_defaults=False)
            state["approval_required"] = True
            state["agent_message"] = "Optimization completed. Approval required."
            self._append_trace(state, "optimizer_node: complete")
            return state

        except Exception as e:
            state["last_error"] = f"Optimizer error: {e}"
            self._append_trace(state, f"optimizer_node: exception={e}")
            state["agent_message"] = "I hit an error while optimizing. Please retry."
            return state

    # ==========================================================
    # NODE 4: Approval
    # ==========================================================
    def approval_node(self, state: TravelState) -> TravelState:
        self._append_trace(state, "approval_node: start")

        approval_mode = state.get("approval_mode", "auto")
        decision = state.get("approval_decision")

        if approval_mode == "auto":
            state["approval_required"] = False
            state["approval_decision"] = "approved"
            state["agent_message"] = "Auto-approval enabled. Plan marked approved."
            self._append_trace(state, "approval_node: auto-approved")
            return state

        # Manual mode
        if decision not in ("approved", "rejected"):
            state["approval_required"] = True
            state["agent_message"] = "Please approve or reject the plan."
            self._append_trace(state, "approval_node: awaiting decision")
            return state

        if decision == "approved":
            state["approval_required"] = False
            state["agent_message"] = "Plan approved by user."
            self._append_trace(state, "approval_node: approved")
            return state

        # rejected -> replan
        fb = state.get("feedback") or "User rejected the plan."
        state["approval_required"] = True
        state["agent_message"] = f"Plan rejected. Feedback: {fb}"
        self._append_trace(state, "approval_node: rejected -> replan")
        return state

    # ==========================================================
    # NODE 5: Completion
    # ==========================================================
    def completion_node(self, state: TravelState) -> TravelState:
        self._append_trace(state, "completion_node: start")
        state["agent_message"] = "Workflow completed."
        self._append_trace(state, "completion_node: complete")
        return state

    # ==========================================================
    # NODE 6: Error Recovery
    # ==========================================================
    def error_recovery_node(self, state: TravelState) -> TravelState:
        err = state.get("last_error") or "Unknown error."
        state["agent_message"] = f"An error occurred. Please retry. Error: {err}"
        self._append_trace(state, "error_recovery_node: invoked")
        return state


# Export node callables
_agents = TravelAgents()
info_collector = _agents.info_collector_node
planner = _agents.planner_node
optimizer = _agents.optimizer_node
approval = _agents.approval_node
completion = _agents.completion_node
error_recovery = _agents.error_recovery_node
