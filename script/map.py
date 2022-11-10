import json
import os
from datetime import datetime

from here_location_services import LS
from here_location_services.config.routing_config import ROUTING_RETURN
from here_map_widget import Map, Marker, GeoJSON
from here_location_services.config.isoline_routing_config import RANGE_TYPE, ISOLINE_ROUTING_TRANSPORT_MODE


def read_api_key():
    """
    reads and parse and store the credential information from local sources
    :return:
    credd : dict
        a dictionary storing different keys and api keys
    """
    here_path = os.path.join(os.getcwd(), "..")
    here_path = os.path.join(here_path, "credentials")
    cred_path = os.path.join(here_path, "credentials.properties")
    # print(cred_path)
    cred = {}
    with open(cred_path, "r") as f:
        for temp in f:
            temp = (temp.strip()).split("=")
            # print(temp)
            cred[temp[0].strip()] = temp[1].strip()

    return cred


class MapAPI:
    def __init__(self):
        # load credentials
        self.creds = read_api_key()
        self.LS_API_KEY = self.creds["here.api_key"]
        self.GMAP_API_KEY = self.creds["gmap.api_key"]
        self.ls = LS(api_key=self.LS_API_KEY)
        self.route_map = None

    def address_to_wgs84(self, address_or_zipcode):
        """
        convert given address code into WGS84 format: longtitude and latitude
        :param address_or_zipcode: string
            address of the location to be converted
        :return geo: JSON object of the address which contains information including WGS
        """
        import requests
        # lat,lng = None, None
        results = None
        api_key = self.GMAP_API_KEY
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        endpoint = f"{base_url}?address={address_or_zipcode}&key={api_key}"
        # see how our endpoint includes our API key? Yes this is yet another reason to restrict the key
        r = requests.get(endpoint)
        if r.status_code not in range(200, 299):
            return None
        try:
            '''
            This try block incase any of our inputs are invalid. This is done instead
            of actually writing out handlers for all kinds of responses.
            '''
            results = r.json()['results'][0]
            # lat = results['geometry']['location']['lat']
            # lng = results['geometry']['location']['lng']
        except:
            pass
        print(results)
        return results

    def create_map(self, center, zoom):
        """

        :param center: [float, float] a lontitude and latitude pair to center the map to
        :param zoom: zoom into the map
        :return:
        """
        self.route_map = Map(api_key=self.LS_API_KEY, center=center, zoom=zoom)
    def adjust_map(self):
        self.route_map.
    def calculate_route(self, start=None, destination=None, mode="car", departure_time=datetime.now(), avoidance=None):

        if destination is None:
            destination = [33.0811809, -96.841015]
        if start is None:
            start = [32.9857, -96.7502]

        # via = waypoint - stupid naming

        if mode == "car":
            result = self.ls.car_route(
                origin=start,
                destination=destination,
                via=None,
                alternatives=2,
                departure_time=departure_time,
                return_results=[
                    ROUTING_RETURN.polyline,
                    ROUTING_RETURN.elevation,
                    ROUTING_RETURN.instructions,
                    ROUTING_RETURN.actions,
                ],
                avoid_areas=avoidance
            )
            geo_json = result.to_geojson()
            return geo_json
        else:
            return "Not implemented yet"


"""
Testing
"""
# maps = MapAPI()
# print(maps.calculate_route())
# results = maps.address_to_wgs84("Toyota North America")
# lat = results['geometry']['location']['lat']
# lng = results['geometry']['location']['lng']
# name = []
# for i in range(0,len(results['address_components'][0])):
#     name.append(results['address_components'][i]['long_name'])
# print(lat,lng,name)
