"""
Weather Tool — Real-time weather data via OpenWeatherMap API.
Returns current conditions and a basic forecast for any city.
"""

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from backend.config import get_settings

settings = get_settings()

# OpenWeatherMap API base URL
_OWM_BASE = "https://api.openweathermap.org/data/2.5"

# Weather condition code → emoji
_CONDITION_EMOJI = {
    "Clear": "☀️",
    "Clouds": "☁️",
    "Rain": "🌧️",
    "Drizzle": "🌦️",
    "Thunderstorm": "⛈️",
    "Snow": "❄️",
    "Mist": "🌫️",
    "Smoke": "🌫️",
    "Haze": "🌫️",
    "Fog": "🌫️",
}


class WeatherInput(BaseModel):
    city: str = Field(description="City name, e.g. 'London', 'Mumbai', 'New York'")
    units: str = Field(
        default="metric",
        description="Temperature units: 'metric' (Celsius), 'imperial' (Fahrenheit), or 'standard' (Kelvin)",
    )


@tool("get_weather", args_schema=WeatherInput)
def get_weather_tool(city: str, units: str = "metric") -> str:
    """
    Get the current weather conditions for any city in the world.
    Returns temperature, humidity, wind speed, and a description.
    Use this when the user asks about weather in a specific location.

    Args:
        city: Name of the city
        units: 'metric' for Celsius (default), 'imperial' for Fahrenheit

    Returns:
        Formatted weather report string
    """
    api_key = settings.openweather_api_key if hasattr(settings, "openweather_api_key") else ""

    if not api_key:
        return (
            "Weather tool is not configured. "
            "Please set OPENWEATHER_API_KEY in your .env file. "
            "Get a free key at: https://openweathermap.org/api"
        )

    unit_symbol = "°C" if units == "metric" else ("°F" if units == "imperial" else "K")

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"{_OWM_BASE}/weather",
                params={
                    "q": city,
                    "appid": api_key,
                    "units": units,
                },
            )

        if response.status_code == 404:
            return f"City '{city}' not found. Please check the city name and try again."
        elif response.status_code == 401:
            return "Invalid OpenWeatherMap API key. Please check OPENWEATHER_API_KEY in .env"
        elif response.status_code != 200:
            return f"Weather API error (HTTP {response.status_code}): {response.text}"

        data = response.json()

        # Extract data
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        wind_unit = "m/s" if units == "metric" else "mph"
        description = data["weather"][0]["description"].capitalize()
        condition_main = data["weather"][0]["main"]
        emoji = _CONDITION_EMOJI.get(condition_main, "🌡️")
        country = data["sys"]["country"]
        visibility = data.get("visibility", 0) / 1000  # convert m → km

        return (
            f"{emoji} Weather in {city.title()}, {country}:\n"
            f"  Condition: {description}\n"
            f"  Temperature: {temp}{unit_symbol} (feels like {feels_like}{unit_symbol})\n"
            f"  Humidity: {humidity}%\n"
            f"  Wind Speed: {wind_speed} {wind_unit}\n"
            f"  Visibility: {visibility:.1f} km"
        )

    except httpx.TimeoutException:
        return f"Weather API timed out. Please try again."
    except Exception as e:
        return f"Failed to fetch weather for '{city}': {str(e)}"
