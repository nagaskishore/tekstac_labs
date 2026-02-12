
"""
Amadeus Experience Search Toolkit
---------------------------------
Provides a class to search for experiences/activities using the Amadeus API.
Class and methods are documented for agent/tool integration.
"""

from amadeus import Client, ResponseError
import os
from typing import List, Dict, Any, Optional
try:
    from config import AMADEUS_CLIENT_ID, AMADEUS_CLIENT_SECRET
except ImportError:
    # Fallback for when imported directly
    from dotenv import load_dotenv
    load_dotenv()
    AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
    AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")

class AmadeusExperienceToolkit:
    """
    Toolkit for searching experiences/activities using Amadeus API.
    Encapsulates Amadeus client and provides agent-friendly methods.
    """
    def __init__(self):
        client_id = AMADEUS_CLIENT_ID
        client_secret = AMADEUS_CLIENT_SECRET
        
        if not client_id or not client_secret:
            raise ValueError("Amadeus API credentials required. Set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET in environment or .env file")
        
        self.amadeus = Client(
            client_id=client_id,
            client_secret=client_secret
        )

    def get_city_code(self, city_name: str) -> Optional[str]:
        """
        Get the IATA city code for a given city name using Amadeus API.
        Args:
            city_name (str): Name of the city.
        Returns:
            Optional[str]: IATA city code if found, else None.
        """
        try:
            response = self.amadeus.reference_data.locations.get(keyword=city_name, subType='CITY')
            locations = response.data
            return locations[0]['iataCode'] if locations else None
        except ResponseError as error:
            print(f"City code failed: {error}")
            return None

    def experience_search(self, city_name: str, radius_km: int = 20, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for experiences/activities in a city using Amadeus API.
        Args:
            city_name (str): Name of the city.
            radius_km (int): Search radius in kilometers (default: 20).
            max_results (int): Maximum number of results to return (default: 10).
        Returns:
            List[Dict[str, Any]]: List of activity details.
        """
        try:
            response = self.amadeus.reference_data.locations.get(keyword=city_name.strip(), subType="CITY")
            if not response.data:
                print(f"City '{city_name}' not found.")
                return []
            city_data = response.data[0]
            geo_code = city_data.get("geoCode", {})
            if not geo_code.get("latitude") or not geo_code.get("longitude"):
                print(f"No coordinates found for '{city_name}'.")
                return []
            activities_resp = self.amadeus.shopping.activities.get(
                latitude=geo_code["latitude"],
                longitude=geo_code["longitude"],
                radius=radius_km
            )
            activities = activities_resp.data if activities_resp and hasattr(activities_resp, 'data') else []
            if not activities:
                print(f"No activities found in {city_name} within {radius_km}km radius.")
                return []
            for act in activities[:max_results]:
                print(f"Name: {act.get('name')}")
                print(f"  Rating: {act.get('rating')}")
                print(f"  Price: {act.get('price', {}).get('amount')} {act.get('price', {}).get('currencyCode')}")
                print(f"  Description: {act.get('shortDescription')}")
                print(f"  Booking: {act.get('bookingLink')}")
                print()
            return activities[:max_results]
        except ResponseError as error:
            print(f"Experience search failed: {error.response.body}")
            return []


if __name__ == "__main__":
    toolkit = AmadeusExperienceToolkit()
    city = input("City for experiences: ").strip()
    radius = input("Search radius in km (default 20): ").strip()
    max_results = input("Max results (default 10): ").strip()
    radius = int(radius) if radius.isdigit() and int(radius) > 0 else 20
    max_results = int(max_results) if max_results.isdigit() and int(max_results) > 0 else 10
    toolkit.experience_search(city, radius_km=radius, max_results=max_results)
