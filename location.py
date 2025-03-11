import requests

def get_location():
    url = "http://ip-api.com/json/"

    try:
        response = requests.get(url)
        result = response.json()

        if result["status"] == "success":
            lat = result["lat"]
            lon = result["lon"]
            city = result["city"]
            country = result["country"]
            map_link = f"https://www.google.com/maps?q={lat},{lon}"

            return {
                "city": city,
                "country": country,
                "latitude": lat,
                "longitude": lon,
                "map_link": map_link
            }
        else:
            return {"error": result}

    except Exception as e:
        return {"error": str(e)}
