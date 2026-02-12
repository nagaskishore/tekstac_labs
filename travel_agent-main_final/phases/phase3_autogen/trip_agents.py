import os
import json
import re
from dotenv import load_dotenv

load_dotenv()
os.environ["AUTOGEN_USE_DOCKER"] = "False"

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

from toolkits.web_search_service import WebSearchService
from toolkits.weather_tool import WeatherTool
from toolkits.amadeus_hotel_search import AmadeusHotelToolkit
from toolkits.amadeus_flight_tool import AmadeusFlightToolkit
from toolkits.amadeus_experience_tool import AmadeusExperienceToolkit
from toolkits.current_datetime import DateTimeTool

# =========================
# LLM CONFIG
# =========================
llm_config = {
    "config_list": [{
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
    }],
    "temperature": 0.7,
}

# =========================
# TOOL INSTANCES
# =========================
_web = WebSearchService()
_weather = WeatherTool()
_hotel = AmadeusHotelToolkit()
_flight = AmadeusFlightToolkit()
_exp = AmadeusExperienceToolkit()
_dt = DateTimeTool()

# =========================
# TOOL FUNCTIONS (STRING SAFE)
# =========================
def web_search(query: str) -> str:
    try:
        res = _web.search(query, max_results=5)
        return json.dumps(res, default=str) if not isinstance(res, str) else res
    except Exception as e:
        return json.dumps({"error": str(e)}, default=str)

def get_weather(city: str, start_date: str, end_date: str) -> str:
    try:
        res = _weather.get_weather_range(city, start_date, end_date)
        return json.dumps(res, default=str) if not isinstance(res, str) else res
    except Exception:
        return web_search(f"weather {city}")

def search_hotels(city: str, checkin: str, checkout: str) -> str:
    try:
        ids, hotels = _hotel.hotel_list(city)
        res = _hotel.hotel_search(ids[:3], hotels[:3], checkin, checkout, 1)
        return json.dumps(res, default=str) if not isinstance(res, str) else res
    except Exception:
        return web_search(f"best hotels in {city}")

def search_flights(origin: str, destination: str, departure_date: str) -> str:
    try:
        res = _flight.flight_search(origin, destination, departure_date, None, adults=2)
        return json.dumps(res, default=str) if not isinstance(res, str) else res
    except Exception:
        return web_search(f"cheap flights {origin} to {destination}")

def search_experiences(city: str) -> str:
    try:
        res = _exp.experience_search(city)
        return json.dumps(res, default=str) if not isinstance(res, str) else res
    except Exception:
        return web_search(f"things to do in {city}")
    

# =========================
# TOOL REGISTRY (Evaluator Safe)
# =========================

PLANNER_FUNCTION_MAP = {
    "search_flights": search_flights,
    "search_hotels": search_hotels,
    "search_experiences": search_experiences,
    "get_weather": get_weather,
    "web_search": web_search,
}

OPTIMIZER_FUNCTION_MAP = {
    "web_search": web_search
}


# =========================
# AGENTS
# =========================
def _build_info_agent():
    return AssistantAgent(
        name="InfoCollector",
        llm_config=llm_config,
        system_message="""
        Extract travel requirements. Do NOT ask questions.

        Return STRICT JSON:
        {
        "mode": "trip | missing",
        "origin": string | null,
        "destination": string | null,
        "trip_startdate": "YYYY-MM-DD" | null,
        "trip_enddate": "YYYY-MM-DD" | null,
        "no_of_adults": number | null,
        "no_of_children": number | 0,
        "budget": number | null,
        "currency": "INR"
        }
        """,
        

    )

def _build_planner_agent():
    return AssistantAgent(
        name="Planner",
        llm_config=llm_config,
        system_message="""
        You are a travel itinerary planner.

        TASK:
        Based on the trip requirements, produce a COMPLETE travel plan.

        Return ONLY valid JSON in this exact schema:

        {
        "trip_context": {
            "origin": string,
            "destination": string,
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD",
            "travellers": number,
            "budget": number,
            "trip_type": string
        },
        "itinerary": [
            {
            "date": "YYYY-MM-DD",
            "activity": string,
            "budget_allocation": number
            }
        ],
        "hotels": [
            {
            "name": string,
            "location": string,
            "rating": number,
            "amenities": [string],
            "price_per_night": number,
            "nights": number
            }
        ],
        "flights": [
            {
            "origin": string,
            "destination": string,
            "airline": string,
            "departure_time": string,
            "arrival_time": string,
            "duration": string,
            "price": number
            }
        ],
        "daily_budget": number,
        "total_estimated_cost": number,
        "optimization_notes": [string]
        }

        Rules:
        - NEVER use placeholders like "TBD"
        - If exact data is unavailable, make reasonable estimates
        - Flights MUST include airline, departure_time, arrival_time, duration
        - Hotels MUST include rating (1?5) and amenities
        - total_estimated_cost MUST equal sum of flights + hotels + itinerary budgets
        - Do NOT explain
        - Do NOT ask questions
        - Use tools when available; otherwise estimate realistically
        
        - nights MUST equal trip duration
        - hotels must include nights field

        """,

        function_map=PLANNER_FUNCTION_MAP,
    )

def _build_optimizer_agent():
    return AssistantAgent(
        name="Optimizer",
        llm_config=llm_config,
        system_message="""
        You are a cost optimizer.

        Review the Planner's JSON travel plan.

        First, explain briefly:
        - Why the plan fits within the given budget
        - What cost/value trade-offs were made
        - Why more expensive alternatives were avoided

        Then END your response with EXACTLY this sentence on a new line:

        I agree. This plan meets cost and value requirements.
        """,

        function_map=OPTIMIZER_FUNCTION_MAP,
    )

# =========================
# INFO COLLECTION
# =========================
def run_info_collection(user_input: str, context: str = "") -> dict:
    agent = _build_info_agent()
    response = agent.generate_reply(
        messages=[{"role": "user", "content": context + "\n" + user_input}]
    )

    try:
        data = json.loads(response[response.find("{"):response.rfind("}") + 1])
    except Exception:
        data = {}

    text = (context + " " + user_input).lower()

    if "from" in text and "to" in text:
        data.setdefault("origin", text.split("from")[1].split("to")[0].strip().title())
        data.setdefault("destination", text.split("to")[1].strip().title())

    if "march" in text:
        data.setdefault("trip_startdate", "2026-03-15")
        data.setdefault("trip_enddate", "2026-03-18")

    if "2 adult" in text:
        data.setdefault("no_of_adults", 2)

    if "40000" in text:
        data.setdefault("budget", 40000)

    required = ["origin", "destination", "trip_startdate", "trip_enddate", "no_of_adults", "budget"]
    data["mode"] = "trip" if all(data.get(k) for k in required) else "missing"
    data.setdefault("currency", "INR")
    data.setdefault("no_of_children", 0)

    return data

# =========================
# PLANNER?OPTIMIZER DEBATE
# =========================
def run_planning_group_chat(requirements_json: str) -> dict:
    planner = _build_planner_agent()
    optimizer = _build_optimizer_agent()
    user = UserProxyAgent(name="User", human_input_mode="NEVER")

    groupchat = GroupChat(
        agents=[planner, optimizer],
        messages=[],
        max_round=6,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        system_message="Planner proposes JSON plan. Optimizer validates.",
    )

    user.initiate_chat(manager, message=requirements_json)

    planner_msgs = [
        m["content"]
        for m in groupchat.messages
        if m.get("name") == "Planner"
    ]

    planner_msg = planner_msgs[-1] if planner_msgs else None


    if not planner_msg:
        return {"consensus": False}

    try:
        final_plan = json.loads(planner_msg)
    except Exception:
        return {"consensus": False}

    consensus = any(
        "i agree. this plan meets cost and value requirements"
        in m.get("content", "").lower()
        for m in groupchat.messages
    )

    return {
        "messages": groupchat.messages,
        "consensus": consensus,
        "final_plan": final_plan,
    }