
"""
Amadeus Hotel Search Toolkit
----------------------------
Provides a class to search for hotels and offers using the Amadeus API.
Class and methods are documented for agent/tool integration.
"""

from amadeus import Client, ResponseError
import os
from typing import List, Tuple, Dict, Any, Optional
try:
    from ..config import AMADEUS_CLIENT_ID, AMADEUS_CLIENT_SECRET
except ImportError:
    # Fallback for when imported directly
    from dotenv import load_dotenv
    load_dotenv()
    AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
    AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")

class AmadeusHotelToolkit:
    """
    Toolkit for searching hotels and offers using Amadeus API.
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

    def hotel_list(self, city_name: str, radius: int = 5) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Get a list of hotel IDs and hotel details for a city using Amadeus API.
        Args:
            city_name (str): Name of the city.
            radius (int): Search radius in KM (default: 5).
        Returns:
            Tuple[List[str], List[Dict[str, Any]]]: List of hotel IDs and hotel details.
        """
        city_code = self.get_city_code(city_name)
        if not city_code:
            print(f"City '{city_name}' not found.")
            return [], []
        try:
            response = self.amadeus.reference_data.locations.hotels.by_city.get(cityCode=city_code, radius=radius, radiusUnit="KM")
            hotels = response.data
            hotel_ids = [hotel.get("hotelId") for hotel in hotels]
            return hotel_ids, hotels
        except ResponseError as error:
            print(f"Hotel list failed: {error.response.body}")
            return [], []

    def extract_hotel_info(self, offer: dict) -> None:
        """
        Print offer details for a hotel (does not print hotel info).
        Args:
            offer (dict): Offer data for a hotel from Amadeus API.
        Returns:
            None
        """
        if 'offers' in offer:
            for offer_item in offer.get('offers', []):
                print("--- Offer Info ---")
                print(f"  Offer ID: {offer_item.get('id', 'N/A')}")
                print(f"  Check-in: {offer_item.get('checkInDate', 'N/A')}")
                print(f"  Check-out: {offer_item.get('checkOutDate', 'N/A')}")
                print(f"  Rate Code: {offer_item.get('rateCode', 'N/A')}")
                print(f"  Board Type: {offer_item.get('boardType', 'N/A')}")
                print(f"  Room Type: {offer_item.get('roomType', 'N/A')}")
                print(f"  Room Type Code: {offer_item.get('roomTypeCode', 'N/A')}")
                desc = offer_item.get('description', {}).get('text')
                if desc:
                    print(f"  Description: {desc}")
                room = offer_item.get('room', {})
                print(f"  Room: {room.get('type', 'N/A')}")
                print(f"  Room Category: {room.get('category', 'N/A')}")
                print(f"  Beds: {room.get('beds', 'N/A')}")
                print(f"  Bed Type: {room.get('bedType', 'N/A')}")
                room_desc = room.get('description', {}).get('text')
                if room_desc:
                    print(f"  Room description: {room_desc}")
                room_name = room.get('name', {}).get('text')
                if room_name:
                    print(f"  Room name: {room_name}")
                guests = offer_item.get('guests', {})
                print(f"  Adults: {guests.get('adults', 'N/A')}")
                print(f"  Children: {guests.get('children', 'N/A')}")
                price = offer_item.get('price', {})
                print(f"  Price: {price.get('total', 'N/A')} {price.get('currency', 'N/A')}")
                print(f"  Base Price: {price.get('base', 'N/A')} {price.get('currency', 'N/A')}")
                print(f"  Taxes: {price.get('taxes', 'N/A')}")
                variations = price.get('variations', {})
                avg_base = variations.get('average', {}).get('base')
                if avg_base:
                    print(f"  Average base price: {avg_base}")
                changes = variations.get('changes', [])
                for change in changes:
                    start = change.get('startDate')
                    end = change.get('endDate')
                    base = change.get('base')
                    if start and end and base:
                        print(f"  Price change: {base} ({start} to {end})")
                policies = offer_item.get('policies', {})
                refundable = policies.get('refundable', {}).get('cancellationRefund')
                if refundable:
                    print(f"  Refund policy: {refundable}")
                cancellation = policies.get('cancellation', {})
                if cancellation:
                    print(f"  Cancellation policy: {cancellation}")
                breakfast = offer_item.get('breakfast', {})
                if breakfast:
                    print(f"  Breakfast: {breakfast}")
                if 'self' in offer_item:
                    print(f"  Offer link: {offer_item['self']}")
                print()

    def hotel_search(

        self,
        hotel_ids: list,
        hotels: list,
        check_in_date: str,
        check_out_date: str,
        adults: int = 1,

        verbose: bool = False,   # ? NEW

        max_hotels: int = 5      # ? NEW: limit returned payload for agents

    ) -> list:

        """

        Search for hotel offers for a list of hotel IDs.
 
        Returns:

            List[Dict]: simplified offers for agent usage

        """

        try:

            if not hotel_ids:
                return []
            id_string = ",".join(hotel_ids)
            response = self.amadeus.shopping.hotel_offers_search.get(
                hotelIds=id_string,
                checkInDate=check_in_date,
                checkOutDate=check_out_date,
                adults=adults
                )
 
            # ? CRITICAL: handle None safely

            offers = (response.data or []) if response and hasattr(response, "data") else []

            # Build hotelId -> hotel metadata map (safer than index alignment)

            hotel_meta_map = {}
            for h in (hotels or []):
                hid = h.get("hotelId") or h.get("hotel", {}).get("hotelId")
                if hid:
                    hotel_meta_map[hid] = h
 
            # Build hotelId -> offers list

            hotel_offers_map = {}
            for offer in offers:
                hid = offer.get("hotel", {}).get("hotelId")
                if hid:
                    hotel_offers_map.setdefault(hid, []).append(offer)
 
            # Return simplified list for agents

            simplified: List[Dict[str, Any]] = []
            for hid in hotel_ids[:max_hotels]:
                meta = hotel_meta_map.get(hid, {})
                name = meta.get("name") or meta.get("hotel", {}).get("name")
 
                # pick cheapest offer if available

                offers_for_hotel = hotel_offers_map.get(hid, [])
                cheapest_total = None
                currency = None
                for off in offers_for_hotel:
                    for oi in off.get("offers", []) or []:
                        price = oi.get("price", {}) or {}
                        total = price.get("total")
                        cur = price.get("currency")
                        try:
                            total_f = float(total) if total is not None else None
                        except Exception:
                            total_f = None
                        if total_f is not None and (cheapest_total is None or total_f < cheapest_total):
                            cheapest_total = total_f
                            currency = cur

                simplified.append({
                    "hotelId": hid,
                    "name": name,
                    "city": (meta.get("address", {}) or {}).get("cityName"),
                    "country": (meta.get("address", {}) or {}).get("countryCode"),
                    "latitude": (meta.get("geoCode", {}) or {}).get("latitude"),
                    "longitude": (meta.get("geoCode", {}) or {}).get("longitude"),
                    "cheapest_total": cheapest_total,
                    "currency": currency,
                    "offers_count": len(offers_for_hotel)
                })
 
            # Optional debug prints

            if verbose:
                for row in simplified:
                    print("Hotel:", row)
            return simplified
 
        except ResponseError as error:

            # ? No print spam in tool mode; return empty + let orchestrator tool_log capture the error
            return []

 
if __name__ == "__main__":
    toolkit = AmadeusHotelToolkit()
    city_name = "Paris"
    check_in = "2025-12-01"
    check_out = "2025-12-03"
    hotel_ids, hotels = toolkit.hotel_list(city_name)
    hotel_ids = hotel_ids[:10]
    hotels = hotels[:10]
    if hotel_ids:
        toolkit.hotel_search(hotel_ids, hotels, check_in, check_out, 2)
