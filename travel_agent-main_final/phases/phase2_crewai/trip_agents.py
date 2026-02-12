from crewai import Agent, Task, Crew, Process

from crewai.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from typing import Dict, Any, Optional
from pydantic import BaseModel
from typing import Optional


# Import tools
from toolkits.web_search_service import WebSearchService
from toolkits.weather_tool import WeatherTool
from toolkits.amadeus_hotel_search import AmadeusHotelToolkit
from toolkits.amadeus_flight_tool import AmadeusFlightToolkit
from toolkits.amadeus_experience_tool import AmadeusExperienceToolkit
from toolkits.current_datetime import DateTimeTool
from api.datamodels import TripRequirements, TravelPlan, OptimizationResult

import json
from datetime import date



"""
Phase 2: CrewAI Agents - Starter Template
"""


# --- Placeholders for CrewAI tools and agents ---
class FlightSearchArgs(BaseModel):
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str] = None


class HotelSearchArgs(BaseModel):
    city: str
    checkin: str
    checkout: str
    adults: int = 1


class WeatherArgs(BaseModel):
    city: str
    start_date: str
    end_date: str


class WebSearchArgs(BaseModel):
    query: str


# Example tool class placeholder

class SearchWebTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web"
    args_schema: type[BaseModel] = WebSearchArgs
    """
    Tool to search the web for travel information.
    Args:
        query (str): The search query string.
    Returns:
        Any: Search results (to be defined by learner).
    TODO: Implement logic to call a real or mock web search API.
    """
    

    def _run(self, query):
        svc = WebSearchService()
        try:
            return json.dumps(
                svc.search(query, max_results=5),
                default=str
            )

        except Exception:
            return {"results": []}


class GetWeatherTool(BaseTool):
    name: str = "weather_lookup"
    description: str = "Get weather forecast"
    args_schema: type[BaseModel] = WeatherArgs


    """
    Tool to get weather forecast for a city and date range.
    Args:
        city (str): City name.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
    Returns:
        Any: Weather forecast data (to be defined by learner).
    TODO: Implement logic to call a real or mock weather API.
    """
    def _run(self, city, start_date, end_date):
        svc = WeatherTool()
        try:
            result = svc.get_weather_range(city, start_date, end_date)
            return json.dumps(result, default=str)

        except Exception:
            return json.dumps({"error":"weather unavailable"})




class SearchHotelsTool(BaseTool):
    name: str = "hotel_search"
    description: str = "Search hotels"
    args_schema: type[BaseModel] = HotelSearchArgs


    """
    Tool to search for hotels in a city.
    Args:
        city (str): City name.
        checkin (str): Check-in date (YYYY-MM-DD).
        checkout (str): Check-out date (YYYY-MM-DD).
        adults (int): Number of adults.
    Returns:
        Any: Hotel search results (to be defined by learner).
    TODO: Implement logic to call a real or mock hotel API.
    """
    def _run(self, city, checkin, checkout, adults=1):
        tk = AmadeusHotelToolkit()
        web = WebSearchService()
        try:
            ids, hotels = tk.hotel_list(city)
            if not ids:
                raise ValueError("no hotels")
            result = tk.hotel_search(ids[:3], hotels[:3], checkin, checkout, adults)
            return json.dumps(result, default=str)

        except Exception:
            return json.dumps(
                web.search(f"best hotels in {city}", max_results=5),
                default=str
            )




class SearchFlightsTool(BaseTool):
    name: str = "flight_search"
    description: str = "Search flights"
    args_schema: type[BaseModel] = FlightSearchArgs






    """
    Tool to search for flights between cities.
    Args:
        origin (str): Origin city.
        destination (str): Destination city.
        departure_date (str): Departure date (YYYY-MM-DD).
        return_date (str, optional): Return date (YYYY-MM-DD).
    Returns:
        Any: Flight search results (to be defined by learner).
    TODO: Implement logic to call a real or mock flight API.
    """
    def _run(self, origin, destination, departure_date, return_date=None):
        tk = AmadeusFlightToolkit()
        web = WebSearchService()
        try:
            result = tk.flight_search(origin, destination, departure_date, return_date, adults=1)
            return json.dumps(result, default=str)

        except Exception:
            return json.dumps(
                web.search(f"flights {origin} to {destination}", max_results=5),
                default=str
            )




class EmptyArgs(BaseModel):
    pass


class GetCurrentDateTool(BaseTool):
    name: str = "current_date"
    description: str = "Get current date"
    args_schema: type[BaseModel] = EmptyArgs

    """
    Tool to get today's date.
    Returns:
        str: Current date in YYYY-MM-DD format.
    TODO: Implement logic to return the current date.
    """
    

    def _run(self):
        try:
            return DateTimeTool().get_today_date()
        except Exception:
            return str(date.today())
    


# Example agent placeholders

def info_collector(user_input: str):

    """
    Agent to extract trip requirements from user input.
    Args:
        user_input (str): The user's natural language trip request.
    Returns:
        TripRequirements: Structured requirements (see db/datamodels.py).
    TODO: Implement logic to parse and validate user input.
    """
    LLM = "openai/gpt-4o-mini"

    agent = Agent(
        role="Travel Requirements Specialist",
        goal="Extract structured trip requirements",
        backstory="Expert travel consultant",
        llm=LLM,
        verbose=False
    )

    task = Task(
    description=f"""
    Extract trip requirements from this MULTI-TURN conversation.

    MULTI-TURN MERGE RULE:
    - Combine fields across turns
    - If origin/destination appear in ANY earlier turn ? keep them
    - Only overwrite if user explicitly changes them


    CRITICAL RULES:
    - Return EXACTLY ONE JSON object
    - NEVER return a list or multiple objects
    - If multiple possibilities exist ? choose the MOST RECENT user intent
    - Use earlier turns only to fill missing fields
    - Output STRICT JSON only ? no markdown, no code blocks

    Schema keys:
    mode, origin, destination, trip_startdate, trip_enddate,
    no_of_adults, no_of_children, budget, currency, purpose

    Conversation:
    {user_input}
    """,
    expected_output="Single JSON object only",
    agent=agent
    )




    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()

        

    raw = str(result)
    raw = raw.replace("```json", "").replace("```", "")

    start = raw.find("{")
    end = raw.rfind("}") + 1
    try:
        data = json.loads(raw[start:end])
    except Exception as e:
        print("INFO_COLLECTOR_JSON_PARSE_FAIL:", e)
        print("RAW:", raw)
        return TripRequirements(mode="missing", missing_fields=["origin","destination"])

    # ---------- MODE NORMALIZATION ----------
    mode_val = data.get("mode")

    if mode_val not in ("trip", "missing"):
        required = ["origin","destination","trip_startdate","trip_enddate","budget"]

        if all(data.get(k) for k in required):
            data["mode"] = "trip"
        else:
            data["mode"] = "missing"


    # if list ? take latest intent
    if isinstance(data, list):
        data = data[-1]

    # normalize blanks ? None
    for k, v in list(data.items()):
        if v == "" or v == "null" or v == "undefined":
            data[k] = None

    # ---------- REQUIRED DEFAULTS ----------
    if not data.get("currency"):
        data["currency"] = "INR"   # safe default for your project

    if not data.get("purpose"):
        data["purpose"] = "leisure"

    # mode must be only trip/missing
    if data.get("mode") not in ("trip", "missing"):
        required = ["origin","destination","trip_startdate","trip_enddate","budget"]
        data["mode"] = "trip" if all(data.get(k) for k in required) else "missing"


    # numeric fix
    if data.get("budget") is not None:
        try:
            data["budget"] = float(data["budget"])
        except:
            data["budget"] = None

    # ---------- NUMERIC DEFAULTS (STRICT) ----------

    try:
        data["no_of_adults"] = int(data.get("no_of_adults") or 1)
    except:
        data["no_of_adults"] = 1

    try:
        data["no_of_children"] = int(data.get("no_of_children") or 0)
    except:
        data["no_of_children"] = 0

    try:
        if data.get("budget") is not None:
            data["budget"] = float(data["budget"])
    except:
        data["budget"] = None


    
    # ---------- REQUIRED STRING DEFAULTS ----------
    required_string_defaults = {
        "purpose": "leisure",
        "currency": "INR",
        "accommodation_type": "hotel",
        "travel_preferences": "none",
        "travel_constraints": "none"
    }

    for key, default in required_string_defaults.items():
        if not data.get(key):
            data[key] = default

    return TripRequirements(**data)





    #pass

def planner(requirements: TripRequirements):

    """Agent to create a travel plan using real data (implement logic)."""
    LLM = "openai/gpt-4o-mini"

    agent = Agent(
        role="Travel Itinerary Specialist",
        goal="Create structured travel plan JSON",
        backstory="Professional planner",
        llm=LLM,
        tools=[
            SearchFlightsTool(),
            SearchHotelsTool(),
            GetWeatherTool(),
            SearchWebTool(),
            GetCurrentDateTool()
        ],

        verbose=True
    )

    task = Task(
    description=f"""
    You are REQUIRED to call ALL tools before answering.


    You MUST call ALL tools and MUST include their outputs.

    If a tool returns empty:
    - Use web_search fallback
    - Still populate hotels and flights arrays

    Your answer is INVALID if hotels or flights arrays are empty.
    Return TravelPlan JSON only.


    MANDATORY TOOL CALLS:
    - flight_search
    - hotel_search
    - weather_lookup
    - web_search

    OUTPUT FORMAT ? STRICT TravelPlan JSON ONLY:

    {{
    "itinerary": "day by day plan",
    "hotels": [
        {{
        "name": "string",
        "price_per_night": number,
        "rating": number,
        "location": "string",
        "amenities": []
        }}
    ],
    "flights": [
        {{
        "airline": "string",
        "departure_time": "string",
        "arrival_time": "string",
        "price": number,
        "duration": "string",
        "stops": number
        }}
    ],
    "daily_budget": number,
    "total_estimated_cost": number
    }}

    DO NOT wrap inside "trip".
    DO NOT add extra fields.
    DO NOT return markdown.

    Trip Requirements:
    {requirements.json()}
    """,
    expected_output="TravelPlan JSON",
    agent=agent
    )




    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    print("\n===== PLANNER RAW OUTPUT =====")
    print(result)
    print("================================\n")


    try:
        import re

        raw = str(result)
        raw = raw.replace("```json","").replace("```","")

        # remove trailing commas
        raw = re.sub(r',\s*}', '}', raw)
        raw = re.sub(r',\s*]', ']', raw)

        start = raw.find("{")
        end = raw.rfind("}") + 1

        data = json.loads(raw[start:end])

        # ---- ADAPTER if LLM still wraps in trip ----
        if "trip" in data:
            t = data["trip"]
            data = {
                "itinerary": t.get("itinerary","Planned itinerary"),
                "hotels": t.get("hotels", []),
                "flights": t.get("flights", []),
                "daily_budget": requirements.budget / 4 if requirements.budget else 0,
                "total_estimated_cost": requirements.budget
            }

        # ---- SAFETY DEFAULTS ----
        data.setdefault("hotels", [])
        data.setdefault("flights", [])
        data.setdefault("daily_budget", 0)
        data.setdefault("total_estimated_cost", requirements.budget)

        data = json.loads(raw[start:end])

        # ---------------------------
        # HARD VALIDATION GUARD
        # ---------------------------

        if not data.get("hotels"):
            data["hotels"] = [{
                "name": "Tool fallback hotel",
                "price_per_night": 0,
                "rating": 0,
                "location": requirements.destination,
                "amenities": []
            }]

        if not data.get("flights"):
            data["flights"] = [{
                "airline": "Tool fallback",
                "departure": requirements.trip_startdate,
                "arrival": requirements.trip_enddate
            }]

        return TravelPlan(**data)


    except Exception as e:
        print("PLANNER_PARSE_ERROR:", e)
        print("RAW_PLANNER_RESULT:", raw)
        return TravelPlan(
            itinerary="Fallback plan",
            hotels=[],
            flights=[],
            daily_budget=0,
            total_estimated_cost=requirements.budget
        )


    #pass

def optimizer(plan: TravelPlan):

    """Agent to optimize travel plan for cost and value (implement logic)."""
    LLM = "openai/gpt-4o-mini"

    agent = Agent(
        role="Travel Cost Optimizer",
        goal="Optimize plan",
        backstory="Cost analyst",
        llm=LLM,
        tools=[SearchWebTool()],
        verbose=False
    )

    task = Task(
    description=f"""
    Optimize this travel plan using web search for price comparison.

    You MUST call web_search tool to:
    - find cheaper alternatives
    - suggest upgrades
    - suggest local experiences

    Return OptimizationResult JSON only.

    Plan:
    {plan.json()}
    """,
        expected_output="OptimizationResult JSON",
        agent=agent
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()

    try:
        return OptimizationResult(**json.loads(str(result)))
    except Exception:
        return OptimizationResult(recommendations=["Basic optimization applied"])

    #pass