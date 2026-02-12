"""
Beginner-Friendly Travel Agent Data Models
==========================================

This file contains all the data models for the TravelMate AI application.
Models are designed to be:
- Easy to understand and use
- Robust with sensible defaults
- Flexible for real-world agent responses
- Self-healing when possible

Key Design Principles:
1. Required fields have clear validation messages
2. Optional fields always have defaults
3. Models can auto-fix common issues
4. Nested models are tolerant of missing data
5. All models have helpful utility methods
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Optional, List, Literal, Dict, Union
from datetime import date, datetime
import json

# =============================================================================
# CORE DATABASE MODELS
# =============================================================================

class User(BaseModel):
    """User model - represents a traveler using the system"""
    id: Optional[int] = None
    name: str
    email: str
    profile: Optional[str] = None
    travel_preferences: Optional[str] = None
    travel_constraints: Optional[str] = None
    created_at: Optional[datetime] = None
    
    @field_validator("email")
    def validate_email(cls, email):
        """Basic email validation to match database constraint"""
        if not email or '@' not in email or '.' not in email:
            raise ValueError("Email must contain @ and . characters")
        return email

    def __str__(self) -> str:
        return f"User({self.name})"

# Example: Approve/Reject request model
class ApprovalRequest(BaseModel):
    trip_id: int
    user_id: int
    approval: bool
    phase: str
    feedback: Optional[str] = None
    thread_id: Optional[str] = None

    
class Trip(BaseModel):
    """
    Trip model - represents a planned trip
    
    This is the main entity that tracks all trip information.
    Required fields are validated, optional fields have sensible defaults.
    """
    # System fields
    id: Optional[int] = None
    user_id: int
    phase: Literal["phase1_langflow", "phase2_crewai", "phase3_autogen", "phase4_langgraph"]
    
    # Trip identification
    title: str = "My Trip"
    
    # Required travel details
    origin: str
    destination: str
    trip_startdate: date
    trip_enddate: date
    
    # Travel preferences with safe defaults
    accommodation_type: Literal[
        "hotel", "resort", "hostel", "apartment", "guesthouse", 
        "luxury", "own_place", "friend_place", "official_accommodation", 
        "budget", "family-friendly", "business", "youth hostel"
    ] = "hotel"
    
    # Traveler details with validation
    no_of_adults: int = Field(default=1, ge=1, description="Must be at least 1 adult")
    no_of_children: int = Field(default=0, ge=0, description="Cannot be negative")
    
    # Budget with validation
    budget: float = Field(default=500.0, ge=0, description="Budget cannot be negative")
    currency: str = "USD"
    
    # Trip status
    trip_status: Literal["draft", "confirmed", "in_progress", "completed", "cancelled"] = "draft"
    
    # Optional details with defaults
    purpose: str = "leisure"
    travel_preferences: str = "none"
    travel_constraints: str = "none"
    
    # System timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("trip_enddate")
    def validate_dates(cls, trip_enddate, info):
        """Ensure end date is after start date"""
        if "trip_startdate" in info.data and trip_enddate <= info.data["trip_startdate"]:
            raise ValueError("Trip end date must be after start date")
        return trip_enddate
    
    @field_validator("trip_startdate")
    def validate_future_date(cls, trip_startdate, info):
        """Ensure trip starts in the future for new trips, but allow existing trips with past dates"""
        from datetime import date
        
        # Check if this is an existing trip (has an id) or a new trip
        trip_id = info.data.get("id")
        
        # Only enforce future date validation for new trips (no id)
        if trip_id is None and trip_startdate <= date.today():
            raise ValueError("Trip start date must be in the future")
        return trip_startdate

    @field_validator("accommodation_type", mode="before")
    def normalize_accommodation_type(cls, value):
        """Map combined accommodation types to database literals"""
        if not value:
            return "hotel"
        
        # Convert to lowercase for easier matching
        value_lower = str(value).lower()
        
        # Mapping for combined values from agents
        accommodation_mapping = {
            "luxury hotel": "luxury",
            "luxury resort": "luxury", 
            "budget hotel": "hotel",
            "boutique hotel": "hotel",
            "business hotel": "hotel",
            "luxury accommodation": "luxury",
            "budget accommodation": "hotel",
            "upscale hotel": "luxury",
            "premium hotel": "luxury",
            "high-end hotel": "luxury",
            "5-star hotel": "luxury",
            "4-star hotel": "hotel",
            "3-star hotel": "hotel",
            "budget hostel": "hostel",
            "luxury hostel": "hostel",
            "vacation rental": "apartment",
            "rental apartment": "apartment",
            "holiday apartment": "apartment",
            "serviced apartment": "apartment",
            "airbnb": "apartment",
            "bed and breakfast": "guesthouse",
            "b&b": "guesthouse",
            "inn": "guesthouse",
            "motel": "hotel",
            "lodge": "hotel",
            "villa": "luxury",
            "mansion": "luxury",
            "penthouse": "luxury",
            "suite": "luxury"
        }
        
        # Check for exact matches first
        if value_lower in accommodation_mapping:
            return accommodation_mapping[value_lower]
        
        # Check for partial matches
        for key, mapped_value in accommodation_mapping.items():
            if key in value_lower:
                return mapped_value
        
        # Check if it's already a valid literal
        valid_types = ["hotel", "resort", "hostel", "apartment", "guesthouse", 
                      "luxury", "own_place", "friend_place", "official_accommodation"]
        if value_lower in valid_types:
            return value_lower
        
        # Default fallback
        return "hotel"
    
    # Utility methods for easy use
    def duration_days(self) -> int:
        """Calculate trip duration in days"""
        return (self.trip_enddate - self.trip_startdate).days + 1
    
    def total_travelers(self) -> int:
        """Get total number of travelers"""
        return self.no_of_adults + self.no_of_children
    
    def daily_budget(self) -> float:
        """Calculate budget per day"""
        return self.budget / self.duration_days() if self.budget > 0 else 0.0
    
    def budget_display(self) -> str:
        """Format budget for display"""
        return f"${self.budget:.0f} {self.currency}"
    
    def travelers_display(self) -> str:
        """Format travelers for display"""
        parts = []
        if self.no_of_adults > 0:
            parts.append(f"{self.no_of_adults} adult{'s' if self.no_of_adults > 1 else ''}")
        if self.no_of_children > 0:
            parts.append(f"{self.no_of_children} child{'ren' if self.no_of_children > 1 else ''}")
        return ", ".join(parts)
    
    def route_display(self) -> str:
        """Format route for display"""
        return f"{self.origin} ? {self.destination}"
    
    def __str__(self) -> str:
        return f"Trip({self.title}: {self.route_display()}, {self.duration_days()} days)"


class ChatHistory(BaseModel):
    """Chat history model - tracks conversations"""
    id: Optional[int] = None
    trip_id: Optional[int] = None  # Can be None for pre-trip chats
    user_id: int
    role: Literal["user", "assistant", "system"]
    phase: Optional[Literal["phase1_langflow", "phase2_crewai", "phase3_autogen", "phase4_langgraph"]] = None
    content: str
    metadata: Optional[str] = None
    sequence_number: Optional[int] = None
    created_at: Optional[datetime] = None

    def __str__(self) -> str:
        return f"Chat({self.role}: {self.content[:50]}...)"


class TripPlanModel(BaseModel):
    """Database model for storing trip plan results"""
    id: Optional[int] = None
    trip_id: int
    itinerary_json: Optional[str] = None  # JSON string of itinerary
    hotels_json: Optional[str] = None     # JSON string of hotel list
    flights_json: Optional[str] = None    # JSON string of flight list
    daily_budget: float = 0.0
    total_estimated_cost: float = 0.0
    generated_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: Literal["draft", "approved", "rejected"] = "draft"
    version: int = 1
    agent_metadata: Optional[str] = None  # JSON for agent-specific data
    
    def to_travel_plan(self) -> "TravelPlan":
        """Convert database model to TravelPlan object"""
        import json
        
        # Parse JSON fields
        itinerary = json.loads(self.itinerary_json) if self.itinerary_json else "Itinerary not available"
        hotels = json.loads(self.hotels_json) if self.hotels_json else []
        flights = json.loads(self.flights_json) if self.flights_json else []
        
        return TravelPlan(
            itinerary=itinerary,
            hotels=hotels,
            flights=flights,
            daily_budget=self.daily_budget,
            total_estimated_cost=self.total_estimated_cost
        )
    
    @classmethod
    def from_travel_plan(cls, travel_plan: "TravelPlan", trip_id: int, version: int = 1) -> "TripPlanModel":
        """Create database model from TravelPlan object"""
        import json
        
        return cls(
            trip_id=trip_id,
            itinerary_json=json.dumps(travel_plan.itinerary),
            hotels_json=json.dumps([h.dict() if hasattr(h, 'dict') else h for h in travel_plan.hotels]),
            flights_json=json.dumps([f.dict() if hasattr(f, 'dict') else f for f in travel_plan.flights]),
            daily_budget=travel_plan.daily_budget,
            total_estimated_cost=travel_plan.total_estimated_cost or 0.0,
            version=version,
            status="draft"
        )
    
    def __str__(self) -> str:
        return f"TripPlan(trip_id={self.trip_id}, v{self.version}, {self.status})"


# =============================================================================
# AGENT WORKFLOW MODELS (Beginner-Friendly)
# =============================================================================

class TripRequirements(BaseModel):
    """
    Model for collecting trip requirements from user input
    
    This model is very tolerant and will try to auto-fix issues:
    - Missing optional fields get defaults
    - Invalid mode gets corrected
    - Incomplete data switches to "missing" mode automatically
    """
    
    # Mode: "trip" for complete data, "missing" for incomplete data
    mode: Literal["trip", "missing"] = "trip"
    
    # Core trip requirements
    origin: Optional[str] = None
    destination: Optional[str] = None
    trip_startdate: Optional[date] = None
    trip_enddate: Optional[date] = None
    
    # Traveler details with safe defaults
    no_of_adults: int = Field(default=1, ge=1)
    no_of_children: int = 0
    
    # Budget details with defaults
    budget: Optional[float] = None
    currency: str = "USD"
    
    # Travel preferences with defaults
    accommodation_type: str = "hotel"
    purpose: str = "leisure"
    travel_preferences: str = "none"
    travel_constraints: str = "none"
    
    # Missing information handling
    error: Optional[Literal["MISSING"]] = None
    missing_fields: Optional[List[str]] = None
    agent_message: Optional[str] = None

    @field_validator("trip_startdate", "trip_enddate", mode="before")
    @classmethod
    def parse_date_fields(cls, v):
        """
        CrewAI/instructor may validate in strict mode, so "YYYY-MM-DD" (str)
        won't automatically coerce to date. Convert strings to datetime.date here.
        Handles:
          - "YYYY-MM-DD"
          - "YYYY-MM-DDTHH:MM:SS"
          - datetime -> date
        """
        if v is None or v == "":
            return None

        # Already a date (but not a datetime)
        if isinstance(v, date) and not isinstance(v, datetime):
            return v

        # datetime -> date
        if isinstance(v, datetime):
            return v.date()

        # str -> date
        if isinstance(v, str):
            s = v.strip()
            try:
                # accepts "YYYY-MM-DDTHH:MM:SS"
                if "T" in s:
                    return datetime.fromisoformat(s).date()
                # accepts "YYYY-MM-DD"
                return date.fromisoformat(s[:10])
            except Exception as e:
                raise ValueError(f"Invalid date format: {v}. Expected YYYY-MM-DD") from e

        raise TypeError(f"Invalid date type: {type(v)}")

    @field_validator("trip_enddate")
    def validate_dates(cls, trip_enddate, info):
        start = info.data.get("trip_startdate")
        mode = info.data.get("mode", "trip")

        if mode == "trip" and trip_enddate and start and trip_enddate <= start:
            raise ValueError("Trip end date must be after start date")
        return trip_enddate

    @field_validator("accommodation_type")
    def normalize_accommodation_type(cls, value):
        """Map combined accommodation types to database literals"""
        if not value:
            return "hotel"
        
        # Convert to lowercase for easier matching
        value_lower = str(value).lower()
        
        # Mapping for combined values from agents
        accommodation_mapping = {
            "luxury hotel": "luxury",
            "luxury resort": "luxury", 
            "budget hotel": "hotel",
            "boutique hotel": "hotel",
            "business hotel": "hotel",
            "luxury accommodation": "luxury",
            "budget accommodation": "hotel",
            "upscale hotel": "luxury",
            "premium hotel": "luxury",
            "high-end hotel": "luxury",
            "5-star hotel": "luxury",
            "4-star hotel": "hotel",
            "3-star hotel": "hotel",
            "budget hostel": "hostel",
            "luxury hostel": "hostel",
            "vacation rental": "apartment",
            "rental apartment": "apartment",
            "holiday apartment": "apartment",
            "serviced apartment": "apartment",
            "airbnb": "apartment",
            "bed and breakfast": "guesthouse",
            "b&b": "guesthouse",
            "inn": "guesthouse",
            "motel": "hotel",
            "lodge": "hotel",
            "villa": "luxury",
            "mansion": "luxury",
            "penthouse": "luxury",
            "suite": "luxury"
        }
        
        # Check for exact matches first
        if value_lower in accommodation_mapping:
            return accommodation_mapping[value_lower]
        
        # Check for partial matches
        for key, mapped_value in accommodation_mapping.items():
            if key in value_lower:
                return mapped_value
        
        # Check if it's already a valid literal
        valid_types = ["hotel", "resort", "hostel", "apartment", "guesthouse", 
                      "luxury", "own_place", "friend_place", "official_accommodation",
                      "budget", "family-friendly", "business", "youth hostel"]
        if value_lower in valid_types:
            return value_lower
        
        # Default fallback
        return "hotel"

    @model_validator(mode="before")
    def auto_fix_and_validate(cls, values):
        """
        Auto-fix common issues and validate requirements
        This makes the model very beginner-friendly
        """
        if not isinstance(values, dict):
            values = {}
        
        # Ensure mode is set
        if "mode" not in values:
            values["mode"] = "trip"
        
        mode = values.get("mode", "trip")
        
        if mode == "trip":
            # Check required fields for trip mode
            required_fields = ["origin", "destination", "trip_startdate", "trip_enddate", "no_of_adults", "budget"]
            missing = []
            
            for field in required_fields:
                if not values.get(field):
                    missing.append(field)
            
            # Auto-switch to missing mode if critical fields missing
            if missing:
                values["mode"] = "missing"
                values["missing_fields"] = missing
                values["error"] = "MISSING"
                values["agent_message"] = f"Please provide: {', '.join(missing)}"
        
        elif mode == "missing":
            # Ensure missing mode has required fields
            if not values.get("missing_fields"):
                values["missing_fields"] = ["origin", "destination"]
            if not values.get("error"):
                values["error"] = "MISSING"
            if not values.get("agent_message"):
                values["agent_message"] = "Additional information needed"
        
        return values

    def is_complete(self) -> bool:
        """Check if requirements are complete for trip creation"""
        return self.mode == "trip" and not self.error
    
    def get_missing_info(self) -> str:
        """Get user-friendly message about missing information"""
        if self.mode == "missing" and self.missing_fields:
            return f"Please provide: {', '.join(self.missing_fields)}"
        return "All information collected"

    def to_trip_dict(self, user_id: int, phase: str, title: str = "My Trip") -> dict:
        """
        Convert to Trip model dictionary for database insertion
        Only works if requirements are complete
        """
        if not self.is_complete():
            raise ValueError(f"Cannot convert incomplete requirements to Trip. Missing: {self.missing_fields}")

        return {
            "user_id": user_id,
            "phase": phase,
            "title": title,
            "origin": self.origin,
            "destination": self.destination,
            "trip_startdate": self.trip_startdate,
            "trip_enddate": self.trip_enddate,
            "accommodation_type": self.accommodation_type,
            "no_of_adults": self.no_of_adults or 1,
            "no_of_children": self.no_of_children,
            "budget": self.budget or 500.0,
            "currency": self.currency,
            "purpose": self.purpose,
            "travel_preferences": self.travel_preferences,
            "travel_constraints": self.travel_constraints,
            "trip_status": "draft"
        }

    def __str__(self) -> str:
        if self.mode == "missing":
            return f"TripRequirements(Missing: {self.missing_fields})"
        return f"TripRequirements({self.origin} ? {self.destination})"


class HotelSuggestion(BaseModel):
    """
    Hotel suggestion - very tolerant of missing data
    All fields have defaults to handle real-world agent responses
    """
    name: str = "Hotel Name Not Available"
    price_per_night: float = 0.0
    rating: float = 0.0
    location: str = "Location TBD"
    amenities: List[str] = []
    
    def price_display(self) -> str:
        """Format price for display"""
        if self.price_per_night > 0:
            return f"${self.price_per_night:.2f}/night"
        return "Price TBD"
    
    def rating_display(self) -> str:
        """Format rating for display"""
        if self.rating > 0:
            return f"{self.rating:.1f}?"
        return "Not rated"
    
    def __str__(self) -> str:
        return f"Hotel({self.name}, {self.price_display()})"


class FlightSuggestion(BaseModel):
    """
    Flight suggestion - very tolerant of missing data
    All required fields have defaults
    """
    airline: str = "Airline TBD"
    departure_time: str = "Time TBD"
    arrival_time: Optional[str] = None
    price: float = 0.0
    duration: str = "Duration TBD"
    stops: int = 0
    
    def price_display(self) -> str:
        """Format price for display"""
        if self.price > 0:
            return f"${self.price:.2f}"
        return "Price TBD"
    
    def stops_display(self) -> str:
        """Format stops for display"""
        if self.stops == 0:
            return "Direct"
        elif self.stops == 1:
            return "1 stop"
        else:
            return f"{self.stops} stops"
    
    def __str__(self) -> str:
        return f"Flight({self.airline}, {self.price_display()}, {self.stops_display()})"


class TravelPlan(BaseModel):
    """
    Complete travel plan - flexible structure for agent responses
    Can handle both string and structured itineraries
    """
    itinerary: Union[str, List[Dict[str, Any]], Dict[str, Any]] = "Itinerary will be provided"
    hotels: List[HotelSuggestion] = []
    flights: List[FlightSuggestion] = []
    daily_budget: float = 0.0
    total_estimated_cost: Optional[float] = None
    
    @field_validator("hotels", mode="before")
    def validate_hotels(cls, v):
        """Convert dict/partial data to HotelSuggestion objects"""
        if not v:
            return []
        
        result = []
        for hotel in v:
            if isinstance(hotel, dict):
                # Create HotelSuggestion with defaults for missing fields
                result.append(HotelSuggestion(**hotel))
            elif isinstance(hotel, HotelSuggestion):
                result.append(hotel)
        return result
    
    @field_validator("flights", mode="before")
    def validate_flights(cls, v):
        """Convert dict/partial data to FlightSuggestion objects"""
        if not v:
            return []
        
        result = []
        for flight in v:
            if isinstance(flight, dict):
                # Create FlightSuggestion with defaults for missing fields
                result.append(FlightSuggestion(**flight))
            elif isinstance(flight, FlightSuggestion):
                result.append(flight)
        return result
    
    def hotel_count(self) -> int:
        """Get number of hotel suggestions"""
        return len(self.hotels)
    
    def flight_count(self) -> int:
        """Get number of flight suggestions"""
        return len(self.flights)
    
    def avg_hotel_price(self) -> float:
        """Calculate average hotel price per night"""
        if not self.hotels:
            return 0.0
        valid_prices = [h.price_per_night for h in self.hotels if h.price_per_night > 0]
        return sum(valid_prices) / len(valid_prices) if valid_prices else 0.0
    
    def itinerary_text(self) -> str:
        """Get itinerary as text regardless of input format"""
        if isinstance(self.itinerary, str):
            return self.itinerary
        elif isinstance(self.itinerary, (list, dict)):
            return json.dumps(self.itinerary, indent=2)
        return "Itinerary not available"
    
    def __str__(self) -> str:
        return f"TravelPlan({self.hotel_count()} hotels, {self.flight_count()} flights)"


class OptimizationResult(BaseModel):
    """
    Optimization results - flexible and user-friendly
    All fields have defaults for robust operation
    """
    recommendations: List[str] = []
    cost_savings: float = 0.0
    value_adds: List[str] = []
    final_plan: str = "Optimization complete"
    approval_required: bool = True
    
    def has_savings(self) -> bool:
        """Check if there are cost savings"""
        return self.cost_savings > 0
    
    def savings_display(self) -> str:
        """Format savings for display"""
        if self.cost_savings > 0:
            return f"${self.cost_savings:.2f}"
        return "No savings identified"
    
    def recommendations_count(self) -> int:
        """Get number of recommendations"""
        return len(self.recommendations)
    
    def summary(self) -> str:
        """Get optimization summary"""
        parts = []
        if self.has_savings():
            parts.append(f"Savings: {self.savings_display()}")
        if self.recommendations_count() > 0:
            parts.append(f"{self.recommendations_count()} recommendations")
        if self.value_adds:
            parts.append(f"{len(self.value_adds)} value-adds")
        return ", ".join(parts) if parts else "Basic optimization applied"
    
    def __str__(self) -> str:
        return f"Optimization({self.summary()})"
    