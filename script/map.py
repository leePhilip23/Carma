import json
import os
from here_location_services import LS


class map:
    def __init__(self):
        # load credentials
        self.creds = self.readAPIKey()
        LS_API_KEY = self.creds["here.api_key"]
        self.GMAP_API_KEY = self.creds["gmap.api_key"]
        self.ls = LS(api_key=LS_API_KEY)

    def addressToWGS84(self, address_or_zipcode):
        """
        convert given address code into WGS84 format: longtitude and latitude
        :param address_or_zipcode: string
            address of the location to be converted
        :return geo: JSON object of the address which contains information including WGS
        """
        import requests

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
        except:
            pass
        return results

    def readAPIKey(self):
        here_path = os.path.join(os.getcwd(), "../creds")
        cred_path = os.path.join(here_path, "credentials.properties")
        print(cred_path)
        cred = {}
        with open(cred_path, "r") as f:
            for temp in f:
                temp = (temp.strip()).split("=")
                print(temp)
                cred[temp[0].strip()] = temp[1].strip()

        return cred


"""
Testing
"""
# maps = map()
# print(json.dumps(maps.addressToWGS84("Toyota North America"), indent=2, sort_keys=True))
