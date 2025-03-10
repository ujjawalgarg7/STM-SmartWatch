import requests

def get_location():
    url = "http://ip-api.com/json/"
    response = requests.get(url)
    data = response.json()
    
    if data["status"] == "success":
        print(f"Location: {data['city']}, {data['country']}")
        print(f"Latitude: {data['lat']}, Longitude: {data['lon']}")
        print(f"ISP: {data['isp']}")
        print(f"Map: https://www.openstreetmap.org/?mlat={data['lat']}&mlon={data['lon']}")
    else:
        print("Error fetching location.")

get_location()




