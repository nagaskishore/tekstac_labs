from datetime import datetime
import pytz
from typing import Optional

class DateTimeTool:
    """Tool for getting current date and time information"""

    def get_current_datetime(self, timezone: Optional[str] = None, format_str: Optional[str] = None):
        """
        Get current date and time information
        
        Args:
            timezone (str, optional): Timezone name (e.g., 'UTC', 'US/Eastern', 'Asia/Kolkata')
            format_str (str, optional): Custom format string (e.g., '%Y-%m-%d %H:%M:%S')
            
        Returns:
            dict: {"datetime": str, "date": str, "time": str, "timestamp": float, "timezone": str}
                  or {"error": str} on failure
        """
        try:
            if timezone:
                if timezone.strip() == "":
                    return {"error": "Timezone cannot be empty string"}
                try:
                    tz = pytz.timezone(timezone.strip())
                    current_dt = datetime.now(tz)
                except pytz.exceptions.UnknownTimeZoneError:
                    return {"error": f"Unknown timezone '{timezone}' - use format like 'UTC', 'US/Eastern', 'Asia/Kolkata'"}
            else:
                current_dt = datetime.now()
            
            result = {
                "datetime": current_dt.isoformat(),
                "date": current_dt.date().isoformat(),
                "time": current_dt.time().isoformat(),
                "timestamp": current_dt.timestamp(),
                "timezone": str(current_dt.tzinfo) if current_dt.tzinfo else "local"
            }
            
            if format_str:
                if not format_str.strip():
                    return {"error": "Format string cannot be empty"}
                try:
                    result["formatted"] = current_dt.strftime(format_str.strip())
                except ValueError as e:
                    return {"error": f"Invalid format string '{format_str}': {str(e)}"}
            
            return result
            
        except Exception as e:
            return {"error": f"DateTime operation failed: {str(e)}"}

    def get_today_date(self):
        """
        Get today's date in YYYY-MM-DD format
        
        Returns:
            dict: {"date": "YYYY-MM-DD"} or {"error": str}
        """
        try:
            return {"date": datetime.now().date().isoformat()}
        except Exception as e:
            return {"error": f"Date operation failed: {str(e)}"}