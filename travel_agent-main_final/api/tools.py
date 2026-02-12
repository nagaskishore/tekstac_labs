from typing import Optional, Dict, Any, List
from .datamodels import HotelSuggestion, FlightSuggestion

from toolkits.amadeus_hotel_search import AmadeusHotelToolkit
from toolkits.amadeus_flight_tool import AmadeusFlightToolkit
from toolkits.amadeus_experience_tool import AmadeusExperienceToolkit
from toolkits.weather_tool import WeatherTool
from toolkits.current_datetime import DateTimeTool
from toolkits.web_search_service import WebSearchService

hotel_toolkit = AmadeusHotelToolkit()
flight_toolkit = AmadeusFlightToolkit()
experience_toolkit = AmadeusExperienceToolkit()
weather_service = WeatherTool()
datetime_service = DateTimeTool()
web_search_service = WebSearchService()

def hotel_search_tool(city: str = "Paris", checkin: str = "2025-12-01", checkout: str = "2025-12-05", adults: int = 1) -> List[HotelSuggestion]:
    hotel_ids, hotels = hotel_toolkit.hotel_list(city)
    if not hotel_ids or not hotels:
        return []
    offers = hotel_toolkit.hotel_search(hotel_ids[:3], hotels[:3], checkin, checkout, adults)
    hotel_suggestions = []
    for idx, offer in enumerate(offers[:3]):
        hotel = hotels[idx] if idx < len(hotels) else {}
        name = hotel.get("name", "Unknown Hotel")
        rating = hotel.get("rating", 0.0)
        address = hotel.get("address", {})
        location = ", ".join(address.get("lines", []))
        amenities = hotel.get("amenities", [])
        price = offer.get("offers", [{}])[0].get("price", {}).get("total", 0.0)
        hotel_suggestions.append(HotelSuggestion(
            name=name,
            price_per_night=price,
            rating=rating,
            location=location,
            amenities=amenities
        ))
    return hotel_suggestions

def flight_search_tool(origin: str = "London", destination: str = "Paris", departure_date: str = "2025-12-01", return_date: Optional[str] = None) -> List[FlightSuggestion]:
    offers = flight_toolkit.flight_search(origin, destination, departure_date, return_date, adults=1)
    flight_suggestions = []
    for offer in offers[:3]:
        price = offer.get('price', {}).get('total', 0.0)
        airline = ', '.join(offer.get('validatingAirlineCodes', [])) if offer.get('validatingAirlineCodes') else 'Unknown Airline'
        itineraries = offer.get('itineraries', [])
        departure_time = ""
        arrival_time = ""
        duration = ""
        stops = 0
        if itineraries:
            first_itin = itineraries[0]
            duration = first_itin.get('duration', "")
            segments = first_itin.get('segments', [])
            if segments:
                departure_time = segments[0].get('departure', {}).get('at', "")
                arrival_time = segments[-1].get('arrival', {}).get('at', "")
                stops = len(segments) - 1
        flight_suggestions.append(FlightSuggestion(
            airline=airline,
            departure_time=departure_time,
            arrival_time=arrival_time,
            price=price,
            duration=duration,
            stops=stops
        ))
    return flight_suggestions

def weather_lookup_tool(city: str = "Paris", start_date: str = "2025-12-01", end_date: str = "2025-12-05") -> Dict[str, Any]:
    result = weather_service.get_weather_range(city, start_date, end_date)
    forecast = result.get("forecast", [])
    if forecast:
        first_day = forecast[0]
        if isinstance(first_day, dict):
            return {
                "date": first_day.get("date", start_date),
                "forecast": first_day.get("description", ""),
                "high": first_day.get("temp_max", None),
                "low": first_day.get("temp_min", None)
            }
        else:
            return {
                "date": start_date,
                "forecast": str(first_day),
                "high": None,
                "low": None
            }
    return {"date": start_date, "forecast": "No data", "high": None, "low": None}

def datetime_tool_func() -> Dict[str, Any]:
    result = datetime_service.get_today_date()
    return {"current_datetime": result.get("date", "") if isinstance(result, dict) else str(result)}

def local_experience_tool(city: str = "Paris") -> List[Dict[str, Any]]:
    # Use AmadeusExperienceToolkit for real experiences if available, else fallback to web search
    try:
        experiences = experience_toolkit.experience_search(city, radius_km=20, max_results=5)
        if experiences:
            return experiences
    except Exception:
        pass
    # fallback to web search
    query = f"things to do in {city}"
    result = web_search_service.search(query, max_results=5)
    experiences = []
    for r in result.get("results", []):
        if isinstance(r, dict):
            name = r.get("title", "Unknown Experience")
        else:
            name = str(r)
        experiences.append({
            "name": name,
            "category": "Activity",
            "price": 0
        })
    return experiences
