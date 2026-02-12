
"""
Amadeus Flight Search Toolkit
-----------------------------
Provides a class to search for flights using the Amadeus API.
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

class AmadeusFlightToolkit:
    """
    Toolkit for searching flights using Amadeus API.
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

        self.CITY_TO_IATA = {
        "bangalore": "BLR",
        "bengaluru": "BLR",
        "goa": "GOI",
        "mumbai": "BOM",
        "delhi": "DEL",
        "hyderabad": "HYD",
        "chennai": "MAA",
        "kolkata": "CCU",
        }
 
    def normalize_city(self,city: str) -> str:
        return (city or "").strip().lower()
 
    def resolve_iata(self,city: str) -> str | None:
        return self.CITY_TO_IATA.get(self.normalize_city(city))
 
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

    def flight_search(
        self,
        origin_city: str,
        dest_city: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Search for flight offers between two cities using Amadeus API.
        Args:
            origin_city (str): Origin city name.
            dest_city (str): Destination city name.
            departure_date (str): Departure date (YYYY-MM-DD).
            return_date (Optional[str]): Return date (YYYY-MM-DD, optional).
            adults (int): Number of adults (default: 1).
        Returns:
            List[Dict[str, Any]]: List of flight offer details.
        """
        origin_code = self.get_city_code(origin_city)
        dest_code = self.get_city_code(dest_city)
        if not origin_code or not dest_code:
            origin_code = self.resolve_iata(origin_city)
            dest_code = self.resolve_iata(dest_city)
            return []
        try:
            search_params = {
                "originLocationCode": origin_code,
                "destinationLocationCode": dest_code,
                "departureDate": departure_date,
                "adults": adults,
                "currencyCode": "USD",
                "max": 4,
            }
            if return_date:
                search_params["returnDate"] = return_date
            response = self.amadeus.shopping.flight_offers_search.get(**search_params)
            offers = response.data if response and hasattr(response, 'data') else []
            if not offers:
                print("No flight offers found.")
                return []
            print("\n--- Parsed Flight Offers ---")
            for offer in offers:
                price = offer.get('price', {})
                print("--- Flight Offer ---")
                print(f"Offer ID: {offer.get('id', 'N/A')}")
                print(f"Type: {offer.get('type', 'N/A')}")
                print(f"Bookable: {offer.get('instantTicketingRequired', 'N/A')}")
                print(f"Validating Airline(s): {', '.join(offer.get('validatingAirlineCodes', []))}")
                print(f"Number of bookable seats: {offer.get('numberOfBookableSeats', 'N/A')}")
                print(f"Total Price: {price.get('total')} {price.get('currency')}")
                print(f"Base Price: {price.get('base', 'N/A')} {price.get('currency', 'N/A')}")
                print(f"Grand Total: {price.get('grandTotal', 'N/A')} {price.get('currency', 'N/A')}")
                print(f"Last Ticketing Date: {offer.get('lastTicketingDate', 'N/A')}")
                print(f"Fare Type: {', '.join(offer.get('pricingOptions', {}).get('fareType', []))}")
                print(f"Included Checked Bags Only: {offer.get('pricingOptions', {}).get('includedCheckedBagsOnly', 'N/A')}")
                print(f"Source: {offer.get('source', 'N/A')}")
                print(f"Self Link: {offer.get('self', 'N/A')}")
                print(f"Number of itineraries: {len(offer.get('itineraries', []))}")
                for itin_idx, itin in enumerate(offer.get('itineraries', []), 1):
                    print(f"  Itinerary {itin_idx}: Duration: {itin.get('duration')}")
                    segments = itin.get('segments', [])
                    print(f"    Number of segments: {len(segments)}")
                    for seg_idx, seg in enumerate(segments, 1):
                        dep = seg.get('departure', {})
                        arr = seg.get('arrival', {})
                        print(f"    Segment {seg_idx}:")
                        print(f"      From: {dep.get('iataCode', 'N/A')} at {dep.get('at', 'N/A')} (Terminal: {dep.get('terminal', 'N/A')})")
                        print(f"      To: {arr.get('iataCode', 'N/A')} at {arr.get('at', 'N/A')} (Terminal: {arr.get('terminal', 'N/A')})")
                        print(f"      Carrier: {seg.get('carrierCode', 'N/A')}")
                        print(f"      Flight Number: {seg.get('number', 'N/A')}")
                        print(f"      Aircraft: {seg.get('aircraft', {}).get('code', 'N/A')}")
                        print(f"      Segment Duration: {seg.get('duration', 'N/A')}")
                        print(f"      Stops: {seg.get('numberOfStops', 'N/A')}")
                        print(f"      Operating Carrier: {seg.get('operating', {}).get('carrierCode', 'N/A')}")
                print(f"Traveler Pricing:")
                for tp in offer.get('travelerPricings', []):
                    print(f"  Traveler ID: {tp.get('travelerId', 'N/A')}, Type: {tp.get('travelerType', 'N/A')}")
                    print(f"  Fare Option: {tp.get('fareOption', 'N/A')}")
                    print(f"  Price: {tp.get('price', {}).get('total', 'N/A')} {tp.get('price', {}).get('currency', 'N/A')}")
                    for fd in tp.get('fareDetailsBySegment', []):
                        print(f"    Segment ID: {fd.get('segmentId', 'N/A')}")
                        print(f"    Cabin: {fd.get('cabin', 'N/A')}, Class: {fd.get('class', 'N/A')}, Fare Basis: {fd.get('fareBasis', 'N/A')}")
                        print(f"    Branded Fare: {fd.get('brandedFareLabel', 'N/A')} ({fd.get('brandedFare', 'N/A')})")
                        print(f"    Included Cabin Bags: {fd.get('includedCabinBags', {}).get('weight', 'N/A')} {fd.get('includedCabinBags', {}).get('weightUnit', 'N/A')}")
                        print(f"    Included Checked Bags: {fd.get('includedCheckedBags', {}).get('weight', 'N/A')} {fd.get('includedCheckedBags', {}).get('weightUnit', 'N/A')}")
                        print(f"    Amenities:")
                        for amenity in fd.get('amenities', []):
                            print(f"      {amenity.get('amenityType', 'N/A')}: {amenity.get('description', 'N/A')} (Chargeable: {amenity.get('isChargeable', 'N/A')})")
                print("-------------------\n")
            return offers
        except ResponseError as error:
            print(f"Flight search failed: {error.response.body}")
            return []

if __name__ == "__main__":
    toolkit = AmadeusFlightToolkit()
    origin = input("Origin city: ").strip()
    dest = input("Destination city: ").strip()
    dep_date = input("Departure date (YYYY-MM-DD): ").strip()
    ret_date = input("Return date (YYYY-MM-DD, optional): ").strip()
    adults = input("Number of adults (default 1): ").strip()
    adults = int(adults) if adults.isdigit() and int(adults) > 0 else 1
    toolkit.flight_search(origin, dest, dep_date, return_date=ret_date if ret_date else None, adults=adults)
