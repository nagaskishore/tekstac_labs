import requests
from datetime import datetime, date

class WeatherTool:
    """Weather forecasting tool using Open-Meteo API"""
    
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    def get_weather(self, place: str, days: int = 3):
        """
        Get weather forecast for specified number of days
        
        Args:
            place (str): Location name (e.g., 'Paris', 'New York')
            days (int): Number of forecast days (1-16)
            
        Returns:
            dict: {"location": {...}, "forecast": [...]} or {"error": str}
        """
        if not place or not place.strip():
            return {"error": "Location name cannot be empty"}
            
        if days < 1 or days > 16:
            return {"error": "Forecast days must be between 1 and 16"}
        
        try:
            # Get coordinates for the place
            coord_params = {"name": place.strip(), "count": 1}
            coord_response = requests.get(self.GEOCODING_URL, params=coord_params, timeout=10)
            
            if not coord_response.ok:
                return {"error": f"Location service unavailable (status: {coord_response.status_code})"}
            
            coord_data = coord_response.json()
            if not coord_data.get("results"):
                return {"error": f"Location '{place}' not found - try a different spelling or nearby city"}
                
            location_info = coord_data["results"][0]
            location = {
                "latitude": location_info["latitude"],
                "longitude": location_info["longitude"],
                "name": location_info["name"],
                "country": location_info.get("country", "Unknown"),
            }
            
            # Get weather forecast
            weather_params = {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                "timezone": "auto",
                "forecast_days": days
            }
            
            weather_response = requests.get(self.WEATHER_URL, params=weather_params, timeout=10)
            
            if not weather_response.ok:
                return {"error": f"Weather service unavailable (status: {weather_response.status_code})"}
            
            weather_data = weather_response.json()
            
            if not weather_data.get("daily", {}).get("time"):
                return {"error": "No weather data available for this location"}
            
            forecast = []
            daily_data = weather_data["daily"]
            
            for i, date_str in enumerate(daily_data["time"]):
                forecast.append({
                    "date": date_str,
                    "temp_max": daily_data["temperature_2m_max"][i],
                    "temp_min": daily_data["temperature_2m_min"][i],
                    "precipitation": daily_data["precipitation_sum"][i],
                })
            
            return {"location": location, "forecast": forecast}
            
        except requests.exceptions.Timeout:
            return {"error": "Weather request timed out - check your internet connection"}
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to weather service - check your internet connection"}
        except Exception as e:
            return {"error": f"Weather lookup failed: {str(e)}"}

    def get_weather_range(self, place: str, start_date: str, end_date: str):
        """
        Get weather forecast for a specific date range
        
        Args:
            place (str): Location name
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: {"location": {...}, "forecast": [...], "date_range": str} or {"error": str}
        """
        if not place or not place.strip():
            return {"error": "Location name cannot be empty"}
            
        if not start_date or not end_date:
            return {"error": "Both start_date and end_date are required"}
        
        # Validate date format and logic
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
            
            if start_dt > end_dt:
                return {"error": "Start date must be before or equal to end date"}
                
            if start_dt < date.today():
                return {"error": "Start date cannot be in the past"}
                
            # Check forecast limit (16 days for free tier)
            days_diff = (end_dt - start_dt).days + 1
            if days_diff > 16:
                return {"error": "Date range too long - maximum 16 days forecast available"}
                
        except ValueError as e:
            return {"error": "Invalid date format - use YYYY-MM-DD (e.g., 2025-06-15)"}
        
        try:
            # Get coordinates
            coord_params = {"name": place.strip(), "count": 1}
            coord_response = requests.get(self.GEOCODING_URL, params=coord_params, timeout=10)
            
            if not coord_response.ok:
                return {"error": f"Location service unavailable (status: {coord_response.status_code})"}
            
            coord_data = coord_response.json()
            if not coord_data.get("results"):
                return {"error": f"Location '{place}' not found - try a different spelling or nearby city"}
                
            location_info = coord_data["results"][0]
            location = {
                "latitude": location_info["latitude"],
                "longitude": location_info["longitude"],
                "name": location_info["name"],
                "country": location_info.get("country", "Unknown"),
            }
            
            # Get weather for date range
            weather_params = {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", 
                         "weather_code", "wind_speed_10m_max"],
                "timezone": "auto",
                "start_date": start_date,
                "end_date": end_date
            }
            
            weather_response = requests.get(self.WEATHER_URL, params=weather_params, timeout=10)
            
            if not weather_response.ok:
                return {"error": f"Weather service unavailable (status: {weather_response.status_code})"}
            
            weather_data = weather_response.json()
            
            if not weather_data.get("daily", {}).get("time"):
                return {"error": f"No weather data available for {place} in the requested date range"}
            
            forecast = []
            daily_data = weather_data["daily"]
            
            for i, date_str in enumerate(daily_data["time"]):
                forecast.append({
                    "date": date_str,
                    "temp_max": daily_data["temperature_2m_max"][i],
                    "temp_min": daily_data["temperature_2m_min"][i],
                    "precipitation": daily_data["precipitation_sum"][i],
                    "weather_code": daily_data["weather_code"][i],
                    "wind_speed_max": daily_data["wind_speed_10m_max"][i],
                })
            
            return {
                "location": location, 
                "forecast": forecast,
                "date_range": f"{start_date} to {end_date}"
            }
            
        except requests.exceptions.Timeout:
            return {"error": "Weather request timed out - check your internet connection"}
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to weather service - check your internet connection"}
        except Exception as e:
            return {"error": f"Weather lookup failed: {str(e)}"}