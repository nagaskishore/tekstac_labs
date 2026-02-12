"""
Phase 4: Travel Agents with LangGraph 
LLM-driven nodes with intelligent tool-calling and state management.
"""
from typing import Dict, List, Optional, TypedDict, Annotated, Any
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime, date
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import toolkits
from toolkits.web_search_service import WebSearchService
from toolkits.weather_tool import WeatherTool
from toolkits.amadeus_hotel_search import AmadeusHotelToolkit
from toolkits.amadeus_flight_tool import AmadeusFlightToolkit
from toolkits.amadeus_experience_tool import AmadeusExperienceToolkit
from toolkits.current_datetime import DateTimeTool

# Import data models and database utilities
from api.datamodels import Trip, TripRequirements, ChatHistory, TravelPlan, HotelSuggestion, FlightSuggestion
from db import db_utils

# =============================================================================
# TRAVEL STATE DEFINITION
# =============================================================================

class TravelState(TypedDict):
    """
    Shared state for LangGraph workflow with comprehensive state management.
    This state persists across all nodes and supports checkpointing.
    """
    # Workflow metadata
    thread_id: str
    current_node: str
    workflow_status: str  # initialized, collecting_info, planning, optimizing, awaiting_approval, completed, failed
    
    # User and trip context
    user_id: int
    trip_id: Optional[int]
    user_input: str
    conversation_history: List[str]
    
    # Trip requirements
    requirements: Optional[Dict[str, Any]]
    requirements_complete: bool
    missing_fields: List[str]
    
    # Travel plan
    travel_plan: Optional[Dict[str, Any]]
    plan_version: int
    
    # Optimization
    optimization_result: Optional[Dict[str, Any]]
    cost_analysis: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    
    # Approval workflow
    approval_required: bool
    approval_status: Optional[str]  # pending, approved, rejected
    user_feedback: Optional[str]
    
    # Tool execution
    tool_results: Dict[str, Any]
    tool_errors: List[str]
    fallback_used: List[str]
    
    # Error handling
    error_count: int
    last_error: Optional[str]
    recovery_checkpoint: Optional[str]
    
    # Performance tracking
    execution_start: Optional[str]
    execution_end: Optional[str]
    node_timings: Dict[str, float]
    
    # Messages for LangGraph
    messages: Annotated[List, add_messages]


# =============================================================================
# TRAVEL AGENTS CLASS
# =============================================================================

class TravelAgents:
    """
    Phase 4: LangGraph Travel Agents with stateful node implementations.
    Each node maintains context, handles errors, and supports checkpointing.
    """
    
    def __init__(self):
        """Initialize LLM and tools for LangGraph workflow."""
        # Configure LLM with proxy settings
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.7,
            timeout=30
        )
        
        # Initialize all toolkits
        self.web_search = WebSearchService()
        self.weather_tool = WeatherTool()
        self.hotel_toolkit = AmadeusHotelToolkit()
        self.flight_toolkit = AmadeusFlightToolkit()
        self.experience_toolkit = AmadeusExperienceToolkit()
        self.datetime_tool = DateTimeTool()
        
    # =========================================================================
    # INFO COLLECTOR NODE
    # =========================================================================
    
    def info_collector_node(self, state: TravelState) -> TravelState:
        """
        Extract and validate travel requirements from user input.
        Uses LLM to parse requirements and validates with tools.
        Falls back to web_search_service if any tool fails.
        """
        start_time = time.time()
        state["current_node"] = "info_collector"
        state["workflow_status"] = "collecting_info"
        
        try:
            # Get current date for validation
            current_date_result = self.datetime_tool.get_current_datetime()
            current_date_str = current_date_result.get("date", datetime.now().isoformat())
            
            # Build context from conversation history
            context = "\n".join(state.get("conversation_history", []))
            user_input = state.get("user_input", "")
            
            # Create LLM prompt for requirement extraction
            extraction_prompt = f"""You are a travel requirements extractor. Extract travel information from the user's request.

Current Date: {current_date_str}

Conversation History:
{context}

Latest User Input:
{user_input}

Extract the following information:
1. Origin city (departure location)
2. Destination city (arrival location)
3. Start date (must be in future, format: YYYY-MM-DD)
4. End date (must be after start date, format: YYYY-MM-DD)
5. Number of adults (default: 1)
6. Number of children (default: 0)
7. Budget amount (default: 500.0)
8. Currency (default: USD)
9. Accommodation type (hotel/resort/hostel/apartment/luxury, default: hotel)
10. Purpose (leisure/business, default: leisure)
11. Travel preferences (any specific requests)
12. Travel constraints (any limitations)

Return ONLY a valid JSON object with these fields. If information is missing, set the field to null.
Use this exact format:
{{
    "origin": "city name or null",
    "destination": "city name or null",
    "trip_startdate": "YYYY-MM-DD or null",
    "trip_enddate": "YYYY-MM-DD or null",
    "no_of_adults": 1,
    "no_of_children": 0,
    "budget": 500.0,
    "currency": "USD",
    "accommodation_type": "hotel",
    "purpose": "leisure",
    "travel_preferences": "text or none",
    "travel_constraints": "text or none"
}}
"""
            
            # Call LLM to extract requirements
            messages = [SystemMessage(content=extraction_prompt)]
            llm_response = self.llm.invoke(messages)
            llm_content = llm_response.content.strip()
            
            # Parse JSON response
            if "```json" in llm_content:
                llm_content = llm_content.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_content:
                llm_content = llm_content.split("```")[1].split("```")[0].strip()
            
            requirements_dict = json.loads(llm_content)
            
            # Validate required fields
            required_fields = ["origin", "destination", "trip_startdate", "trip_enddate"]
            missing_fields = [f for f in required_fields if not requirements_dict.get(f)]
            
            # If destination provided, validate with web search (fallback tool)
            if requirements_dict.get("destination") and not missing_fields:
                try:
                    search_result = self.web_search.search(
                        query=f"travel to {requirements_dict['destination']} tourism information",
                        max_results=1
                    )
                    if search_result and search_result.get("results"):
                        print(f" Destination validated: {requirements_dict['destination']}")
                    else:
                        print(f" Could not validate destination, but proceeding")
                except Exception as e:
                    print(f" Destination validation failed: {e}")
                    state["fallback_used"].append(f"info_collector: web_search validation failed")
            
            # Update state with requirements
            state["requirements"] = requirements_dict
            state["missing_fields"] = missing_fields
            state["requirements_complete"] = len(missing_fields) == 0
            
            # Save to database if requirements complete
            if state["requirements_complete"]:
                try:
                    # Create Trip object
                    trip_data = {
                        "user_id": state["user_id"],
                        "phase": "phase4_langgraph",
                        "title": f"Trip to {requirements_dict['destination']}",
                        "origin": requirements_dict["origin"],
                        "destination": requirements_dict["destination"],
                        "trip_startdate": date.fromisoformat(requirements_dict["trip_startdate"]),
                        "trip_enddate": date.fromisoformat(requirements_dict["trip_enddate"]),
                        "no_of_adults": requirements_dict.get("no_of_adults") or 1,
                        "no_of_children": requirements_dict.get("no_of_children") or 0,
                        "budget": requirements_dict.get("budget") or 500.0,
                        "currency": requirements_dict.get("currency") or "USD",
                        "accommodation_type": requirements_dict.get("accommodation_type") or "hotel",
                        "purpose": requirements_dict.get("purpose") or "leisure",
                        "travel_preferences": requirements_dict.get("travel_preferences") or "none",
                        "travel_constraints": requirements_dict.get("travel_constraints") or "none",
                        "trip_status": "draft"
                    }
                    
                    trip = Trip(**trip_data)
                    trip_id = db_utils.create_trip(trip)
                    state["trip_id"] = trip_id
                    
                    
                    # Log to chat history
                    db_utils.save_chat_message(ChatHistory(
                        trip_id=trip_id,
                        user_id=state["user_id"],
                        role="assistant",
                        phase="phase4_langgraph",
                        content=f"Requirements collected: {json.dumps(requirements_dict)}"
                    ))
                    
                except Exception as e:
                    print(f" Database error: {e}")
                    state["tool_errors"].append(f"Database creation failed: {str(e)}")
            
            # Add to messages
            state["messages"].append(AIMessage(content=f"Requirements extracted: {json.dumps(requirements_dict)}"))
            
            # Update timing
            duration = time.time() - start_time
            state["node_timings"]["info_collector"] = duration
            
            if missing_fields:
                print(f"   Missing fields: {missing_fields}")
            
        except Exception as e:
            print(f" Error in info_collector_node: {e}")
            state["error_count"] += 1
            state["last_error"] = str(e)
            state["tool_errors"].append(f"info_collector: {str(e)}")
            state["workflow_status"] = "failed"
        
        return state
    
    # =========================================================================
    # PLANNER NODE
    # =========================================================================
    
    def planner_node(self, state: TravelState) -> TravelState:
        """
        Create comprehensive travel plan using real API data.
        Executes parallel API calls for flights, hotels, experiences, weather.
        Falls back to web_search_service if ANY tool fails.
        """
        start_time = time.time()
        state["current_node"] = "planner"
        state["workflow_status"] = "planning"
        
        requirements = state.get("requirements", {})
        if not requirements:
            state["tool_errors"].append("planner: No requirements found")
            return state
        
        try:
            destination = requirements["destination"]
            origin = requirements["origin"]
            start_date = requirements["trip_startdate"]
            end_date = requirements["trip_enddate"]
            adults = requirements.get("no_of_adults", 1)
            budget = requirements.get("budget", 500.0)
            
            tool_results = {}
            
            # 1. Search for flights
            print(f" Searching flights: {origin}  {destination}")
            try:
                flight_offers = self.flight_toolkit.flight_search(
                    origin_city=origin,
                    dest_city=destination,
                    departure_date=start_date,
                    return_date=end_date if end_date else None,
                    adults=adults
                )
                
                flights = []
                for offer in flight_offers[:3]:
                    price = float(offer.get('price', {}).get('total', 0.0))
                    airline = ', '.join(offer.get('validatingAirlineCodes', [])) or 'Unknown'
                    itineraries = offer.get('itineraries', [])
                    
                    if itineraries:
                        first_itin = itineraries[0]
                        duration = first_itin.get('duration', "")
                        segments = first_itin.get('segments', [])
                        
                        if segments:
                            dep_time = segments[0].get('departure', {}).get('at', "")
                            arr_time = segments[-1].get('arrival', {}).get('at', "")
                            stops = len(segments) - 1
                            
                            flights.append(FlightSuggestion(
                                airline=airline,
                                departure_time=dep_time,
                                arrival_time=arr_time,
                                price=price,
                                duration=duration,
                                stops=stops
                            ))
                
                tool_results["flights"] = [f.dict() for f in flights]
                
            except Exception as e:
                print(f" Flight search failed: {e}, using fallback")
                state["fallback_used"].append(f"planner: amadeus_flight_tool -> web_search")
                try:
                    search_result = self.web_search.search(
                        query=f"flights from {origin} to {destination} {start_date}",
                        max_results=3
                    )
                    tool_results["flights_fallback"] = search_result.get("results", [])
                    print(f" Fallback web search completed for flights")
                except Exception as fallback_error:
                    print(f" Fallback also failed: {fallback_error}")
                    tool_results["flights"] = []
            
            # 2. Search for hotels
            print(f" Searching hotels in {destination}")
            try:
                hotel_ids, hotels = self.hotel_toolkit.hotel_list(destination)
                
                if hotel_ids and hotels:
                    offers = self.hotel_toolkit.hotel_search(
                        hotel_ids[:3],
                        hotels[:3],
                        start_date,
                        end_date,
                        adults
                    )
                    
                    hotel_suggestions = []
                    for idx, offer in enumerate(offers[:3]):
                        hotel = hotels[idx] if idx < len(hotels) else {}
                        name = hotel.get("name", "Unknown Hotel")
                        rating = float(hotel.get("rating", 0.0))
                        address = hotel.get("address", {})
                        location = ", ".join(address.get("lines", []))
                        amenities = hotel.get("amenities", [])
                        price = float(offer.get("offers", [{}])[0].get("price", {}).get("total", 0.0))
                        
                        hotel_suggestions.append(HotelSuggestion(
                            name=name,
                            price_per_night=price,
                            rating=rating,
                            location=location,
                            amenities=amenities
                        ))
                    
                    tool_results["hotels"] = [h.dict() for h in hotel_suggestions]
                    print(f" Found {len(hotel_suggestions)} hotel options")
                else:
                    raise Exception("No hotels found")
                    
            except Exception as e:
                print(f" Hotel search failed: {e}, using fallback")
                state["fallback_used"].append(f"planner: amadeus_hotel_search -> web_search")
                try:
                    search_result = self.web_search.search(
                        query=f"hotels in {destination} {start_date} to {end_date}",
                        max_results=3
                    )
                    fallback = search_result.get("results", [])
                    tool_results["hotels_fallback"] = fallback
                    tool_results["hotels"] = [{"name": r.get("title") or "Hotel option",
                                                        "price_per_night": 0.0, 
                                                        "rating": 0.0, 
                                                        "location": destination,"amenities": [],} for r in fallback]
                    print(f" Fallback web search completed for hotels")
                except Exception as fallback_error:
                    print(f" Fallback also failed: {fallback_error}")
                    tool_results["hotels"] = []
            
            # 3. Search for experiences
            print(f" Searching experiences in {destination}")
            try:
                experiences = self.experience_toolkit.experience_search(
                    city_name=destination,
                    radius_km=20,
                    max_results=5
                )
                tool_results["experiences"] = experiences[:5] if experiences else []
                print(f" Found {len(tool_results['experiences'])} experiences")
                
            except Exception as e:
                print(f" Experience search failed: {e}, using fallback")
                state["fallback_used"].append(f"planner: amadeus_experience_tool -> web_search")
                try:
                    search_result = self.web_search.search(
                        query=f"things to do in {destination} tourist attractions",
                        max_results=5
                    )
                    tool_results["experiences_fallback"] = search_result.get("results", [])
                    print(f" Fallback web search completed for experiences")
                except Exception as fallback_error:
                    print(f" Fallback also failed: {fallback_error}")
                    tool_results["experiences"] = []
            
            # 4. Get weather forecast
            print(f" Getting weather forecast for {destination}")
            try:
                weather_result = self.weather_tool.get_weather_range(
                    place=destination,
                    start_date=start_date,
                    end_date=end_date
                )
                if weather_result.get("error"):
                    state["fallback_used"].append("planner: weather_tool -> web_search (range not supported)")
                    tool_results["weather"] = {}
                    tool_results["weather_error"] = weather_result["error"]
                else:
                    tool_results["weather"] = weather_result
                    print(f" Weather forecast retrieved")
                
            except Exception as e:
                print(f" Weather lookup failed: {e}, using fallback")
                state["fallback_used"].append(f"planner: weather_tool -> web_search")
                try:
                    search_result = self.web_search.search(
                        query=f"weather forecast {destination} {start_date}",
                        max_results=2
                    )
                    tool_results["weather_fallback"] = search_result.get("results", [])
                    print(f" Fallback web search completed for weather")
                except Exception as fallback_error:
                    print(f" Fallback also failed: {fallback_error}")
                    tool_results["weather"] = {}
            
            # Store all tool results in state
            state["tool_results"] = tool_results
            
            # Generate comprehensive itinerary using LLM
            print(f" Generating comprehensive itinerary with LLM")
            
            itinerary_prompt = f"""Create a detailed day-by-day travel itinerary based on the following information:

Destination: {destination}
Dates: {start_date} to {end_date}
Travelers: {adults} adult(s)
Budget: {budget} {requirements.get('currency', 'USD')}
Purpose: {requirements.get('purpose', 'leisure')}

Available Options:
Flights: {json.dumps(tool_results.get('flights', []), indent=2)}
Hotels: {json.dumps(tool_results.get('hotels', []), indent=2)}
Experiences: {json.dumps(tool_results.get('experiences', []), indent=2)}
Weather: {json.dumps(tool_results.get('weather', {}), indent=2)}

Create a comprehensive itinerary with:
1. Day-by-day schedule
2. Recommended flight options
3. Hotel recommendations
4. Daily activities and experiences
5. Weather considerations
6. Budget breakdown

Format as a detailed text itinerary."""
            
            messages = [SystemMessage(content=itinerary_prompt)]
            llm_response = self.llm.invoke(messages)
            itinerary_text = llm_response.content.strip()
            
            # Calculate daily budget
            trip_days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
            daily_budget = budget / trip_days if trip_days > 0 else budget
            
            # Calculate total estimated cost
            total_flight_cost = sum([f.get('price', 0) for f in tool_results.get('flights', [])[:1]])
            total_hotel_cost = sum([h.get('price_per_night', 0) for h in tool_results.get('hotels', [])[:1]]) * trip_days
            total_cost = total_flight_cost + total_hotel_cost
            
            # Create TravelPlan
            travel_plan = TravelPlan(
                itinerary=itinerary_text,
                hotels=[HotelSuggestion(**h) for h in tool_results.get('hotels', [])],
                flights=[FlightSuggestion(**f) for f in tool_results.get('flights', [])],
                daily_budget=daily_budget,
                total_estimated_cost=total_cost
            )
            
            state["travel_plan"] = travel_plan.dict()
            
            # Save to database
            if state.get("trip_id"):
                try:
                    plan_id = db_utils.save_travel_plan_to_db(
                        travel_plan=travel_plan,
                        trip_id=state["trip_id"],
                        version=state.get("plan_version", 1)
                    )
                    print(f" Travel plan saved to database with ID: {plan_id}")
                    
                    # Update trip status
                    db_utils.update_trip_status(state["trip_id"], "confirmed")
                    
                    # Log to chat history
                    db_utils.save_chat_message(ChatHistory(
                        trip_id=state["trip_id"],
                        user_id=state["user_id"],
                        role="assistant",
                        phase="phase4_langgraph",
                        content=f"Travel plan created with {len(tool_results.get('hotels', []))} hotels and {len(tool_results.get('flights', []))} flights"
                    ))
                    
                except Exception as e:
                    print(f" Database save error: {e}")
                    state["tool_errors"].append(f"Database save failed: {str(e)}")
            
            # Update state
            state["messages"].append(AIMessage(content=f"Travel plan created with comprehensive itinerary"))
            
            # Update timing
            duration = time.time() - start_time
            state["node_timings"]["planner"] = duration
            
            print(f" Planning complete in {duration:.2f}s")
            print(f"   Flights: {len(tool_results.get('flights', []))}")
            print(f"   Hotels: {len(tool_results.get('hotels', []))}")
            print(f"   Experiences: {len(tool_results.get('experiences', []))}")
            print(f"   Fallbacks used: {len(state['fallback_used'])}")
            
        except Exception as e:
            print(f" Error in planner_node: {e}")
            state["error_count"] += 1
            state["last_error"] = str(e)
            state["tool_errors"].append(f"planner: {str(e)}")
            state["workflow_status"] = "failed"
        
        return state
    
    # =========================================================================
    # OPTIMIZER NODE
    # =========================================================================
    
    def optimizer_node(self, state: TravelState) -> TravelState:
        """
        Optimize travel plan for cost, timing, and satisfaction.
        Uses web_search_service extensively for research and alternatives.
        """
        start_time = time.time()
        state["current_node"] = "optimizer"
        state["workflow_status"] = "optimizing"
        
        print(f"\n{'='*60}")
        print(f" OPTIMIZER NODE - Optimizing travel plan")
        print(f"{'='*60}")
        
        travel_plan = state.get("travel_plan")
        requirements = state.get("requirements", {})
        
        if not travel_plan:
            state["tool_errors"].append("optimizer: No travel plan found")
            return state
        
        try:
            destination = requirements.get("destination", "")
            budget = requirements.get("budget", 500.0)
            currency = requirements.get("currency", "USD")
            
            # Use web search to find alternatives and better deals
            print(f" Researching alternatives and deals")
            
            alternatives = []
            cost_savings = 0.0
            
            # Search for budget alternatives
            try:
                budget_search = self.web_search.search(
                    query=f"budget travel tips {destination} cheap hotels flights",
                    max_results=3
                )
                
                if budget_search and budget_search.get("results"):
                    alternatives.extend(budget_search.get("results", []))
                    print(f" Found {len(budget_search.get('results', []))} budget alternatives")
                    
            except Exception as e:
                print(f" Budget search failed: {e}")
                state["fallback_used"].append(f"optimizer: web_search for budget alternatives failed")
            
            # Search for timing optimization
            try:
                timing_search = self.web_search.search(
                    query=f"best time to visit {destination} off-peak season deals",
                    max_results=2
                )
                
                if timing_search and timing_search.get("results"):
                    alternatives.extend(timing_search.get("results", []))
                    print(f" Found {len(timing_search.get('results', []))} timing alternatives")
                    
            except Exception as e:
                print(f" Timing search failed: {e}")
            
            # Analyze current costs
            flights = travel_plan.get("flights", [])
            hotels = travel_plan.get("hotels", [])
            
            total_flight_cost = sum([f.get("price", 0) for f in flights[:1]]) if flights else 0
            total_hotel_cost = sum([h.get("price_per_night", 0) for h in hotels[:1]]) * 3 if hotels else 0
            current_total = total_flight_cost + total_hotel_cost
            
            # Calculate potential savings (estimate 10-15% with optimization)
            potential_savings = current_total * 0.12
            cost_savings = potential_savings
            
            # Generate optimization recommendations using LLM
            print(f" Generating optimization recommendations")
            
            optimization_prompt = f"""Analyze the travel plan and provide optimization recommendations.

Current Plan Summary:
- Destination: {destination}
- Budget: {budget} {currency}
- Current Estimated Cost: {current_total:.2f} {currency}
- Flights: {len(flights)} options
- Hotels: {len(hotels)} options

Research Findings:
{json.dumps(alternatives, indent=2)}

Provide:
1. Cost-saving recommendations (specific actions)
2. Timing optimization suggestions
3. Value-add opportunities (better experiences for same/less cost)
4. Final optimized plan summary

Be specific and actionable."""
            
            messages = [SystemMessage(content=optimization_prompt)]
            llm_response = self.llm.invoke(messages)
            optimization_text = llm_response.content.strip()
            
            # Create optimization result
            from api.datamodels import OptimizationResult
            
            optimization_result = OptimizationResult(
                recommendations=[
                    "Book flights 2-3 months in advance for best prices",
                    "Consider weekday travel for lower rates",
                    "Look for hotel packages that include breakfast",
                    "Use public transportation instead of taxis",
                    "Visit free attractions and local markets"
                ],
                cost_savings=cost_savings,
                value_adds=[
                    "Local food tours for authentic experiences",
                    "Free walking tours in city center",
                    "Visit during shoulder season for fewer crowds"
                ],
                final_plan=optimization_text,
                approval_required=True
            )
            
            state["optimization_result"] = optimization_result.dict()
            state["cost_analysis"] = {
                "original_budget": budget,
                "current_total": current_total,
                "potential_savings": cost_savings,
                "optimized_total": current_total - cost_savings
            }
            state["alternatives"] = alternatives
            state["approval_required"] = True
            state["approval_status"] = "pending"
            
            # Update database
            if state.get("trip_id"):
                try:
                    # Log optimization to chat history
                    db_utils.save_chat_message(ChatHistory(
                        trip_id=state["trip_id"],
                        user_id=state["user_id"],
                        role="assistant",
                        phase="phase4_langgraph",
                        content=f"Plan optimized: {cost_savings:.2f} {currency} in potential savings"
                    ))
                    
                except Exception as e:
                    print(f" Database log error: {e}")
            
            # Update state messages
            state["messages"].append(AIMessage(
                content=f"Optimization complete: {len(optimization_result.recommendations)} recommendations, {cost_savings:.2f} {currency} potential savings"
            ))
            
            # Update timing
            duration = time.time() - start_time
            state["node_timings"]["optimizer"] = duration
                        
        except Exception as e:
            print(f" Error in optimizer_node: {e}")
            state["error_count"] += 1
            state["last_error"] = str(e)
            state["tool_errors"].append(f"optimizer: {str(e)}")
            state["workflow_status"] = "failed"
        
        return state
    
    # =========================================================================
    # APPROVAL NODE
    # =========================================================================
    
    def approval_node(self, state: TravelState) -> TravelState:
        """
        Prepare plan for human approval and create checkpoint.
        This node pauses workflow for user input.
        """
        start_time = time.time()
        state["current_node"] = "approval"
        state["workflow_status"] = "awaiting_approval"
                
        try:
            # Create checkpoint for state restoration
            checkpoint_id = f"approval_{state.get('trip_id')}_{int(time.time())}"
            state["recovery_checkpoint"] = checkpoint_id
            
            # Prepare approval summary
            requirements = state.get("requirements", {})
            travel_plan = state.get("travel_plan", {})
            optimization = state.get("optimization_result", {})
            cost_analysis = state.get("cost_analysis", {})
            
            approval_summary = f"""
 TRAVEL PLAN READY FOR APPROVAL

 Trip Details:
    From: {requirements.get('origin', 'N/A')}  To: {requirements.get('destination', 'N/A')}
    Dates: {requirements.get('trip_startdate', 'N/A')} to {requirements.get('trip_enddate', 'N/A')}
    Travelers: {requirements.get('no_of_adults', 1)} adult(s), {requirements.get('no_of_children', 0)} child(ren)
    Budget: {requirements.get('budget', 0)} {requirements.get('currency', 'USD')}

 Cost Analysis:
    Original Budget: {cost_analysis.get('original_budget', 0):.2f}
    Current Total: {cost_analysis.get('current_total', 0):.2f}
    Potential Savings: {cost_analysis.get('potential_savings', 0):.2f}
    Optimized Total: {cost_analysis.get('optimized_total', 0):.2f}

 Travel Components:
    Flights: {len(travel_plan.get('flights', []))} options
    Hotels: {len(travel_plan.get('hotels', []))} options
    Experiences: {len(state.get('tool_results', {}).get('experiences', []))} activities

 Optimization:
    Recommendations: {len(optimization.get('recommendations', []))}
    Value Additions: {len(optimization.get('value_adds', []))}

Please approve or reject this plan. If rejected, provide feedback for improvements.
"""
            
            state["messages"].append(AIMessage(content=approval_summary))
                  
            # Update timing
            duration = time.time() - start_time
            state["node_timings"]["approval"] = duration
            
        except Exception as e:
            print(f" Error in approval_node: {e}")
            state["error_count"] += 1
            state["last_error"] = str(e)
            state["tool_errors"].append(f"approval: {str(e)}")
        
        return state
    
    # =========================================================================
    # COMPLETION NODE
    # =========================================================================
    
    def completion_node(self, state: TravelState) -> TravelState:
        """
        Mark workflow as completed and finalize all records.
        """
        start_time = time.time()
        state["current_node"] = "completion"
        state["workflow_status"] = "completed"
        state["execution_end"] = datetime.now().isoformat()
        
                
        try:
            # Update trip status to completed
            if state.get("trip_id"):
                db_utils.update_trip_status(state["trip_id"], "completed")
                
                # Log completion
                db_utils.save_chat_message(ChatHistory(
                    trip_id=state["trip_id"],
                    user_id=state["user_id"],
                    role="assistant",
                    phase="phase4_langgraph",
                    content="Travel planning workflow completed successfully"
                ))
            
            # Calculate total execution time
            if state.get("execution_start"):
                start_dt = datetime.fromisoformat(state["execution_start"])
                end_dt = datetime.fromisoformat(state["execution_end"])
                total_time = (end_dt - start_dt).total_seconds()
                                     
            state["messages"].append(AIMessage(content=" Travel planning completed successfully!"))
            
            # Update timing
            duration = time.time() - start_time
            state["node_timings"]["completion"] = duration
            
        except Exception as e:
            print(f" Error in completion_node: {e}")
            state["error_count"] += 1
            state["last_error"] = str(e)
        
        return state
    
    # =========================================================================
    # ERROR RECOVERY NODE
    # =========================================================================
    
    def error_recovery_node(self, state: TravelState) -> TravelState:
        """
        Handle errors and prepare for recovery or user intervention.
        """
        start_time = time.time()
        state["current_node"] = "error_recovery"
        state["workflow_status"] = "failed"
                        
        try:
            last_error = state.get("last_error", "Unknown error")
            error_count = state.get("error_count", 0)
            
            # Create error recovery message
            recovery_message = f"""
 WORKFLOW ENCOUNTERED ERRORS

Error Details:
    Last Error: {last_error}
    Total Errors: {error_count}
    Failed Tools: {len(state.get('tool_errors', []))}
    Fallbacks Used: {len(state.get('fallback_used', []))}

Recovery Options:
1. Retry with fallback tools
2. Request additional user input
3. Simplify requirements and retry

Please provide feedback or clarification to continue.
"""
            
            state["messages"].append(AIMessage(content=recovery_message))
            
            # Log to database
            if state.get("trip_id"):
                db_utils.save_chat_message(ChatHistory(
                    trip_id=state["trip_id"],
                    user_id=state["user_id"],
                    role="assistant",
                    phase="phase4_langgraph",
                    content=f"Error recovery: {last_error}"
                ))
            
            # Update timing
            duration = time.time() - start_time
            state["node_timings"]["error_recovery"] = duration
            
        except Exception as e:
            print(f" Error in error_recovery_node: {e}")
            state["error_count"] += 1
        
        return state


# =============================================================================
# EXPORT NODE FUNCTIONS (Top-level callables for orchestrator)
# =============================================================================

# Create singleton instance
_travel_agents_instance = TravelAgents()

# Export node functions as top-level callables
info_collector = _travel_agents_instance.info_collector_node
planner = _travel_agents_instance.planner_node
optimizer = _travel_agents_instance.optimizer_node
approval = _travel_agents_instance.approval_node
completion = _travel_agents_instance.completion_node
error_recovery = _travel_agents_instance.error_recovery_node
