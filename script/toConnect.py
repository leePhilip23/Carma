from data import DataAPI
from map import MapAPI
from here_location_services.config.matrix_routing_config import AvoidBoundingBox
from math import dist
import pandas as pd
import bz2file as bz2
import pickle

from util import *


class toConnect():
    def __init__(self):
        self.avoidance = []
        self.field_names = ['Weekday_1', 'Weekday_2', 'Weekday_3', 'Weekday_4', 'Weekday_5', 'Weekday_6',
             'Duration', 'Rain', 'Rd', 'St', 'Dr', 'Ave', 'Route',"Pike", 'Fwy', 'Start_Lat', 'Start_Lng',
                            'End_Lat', 'End_Lng', 'Distance(mi)','Humidity(%)']
        self.df = pd.DataFrame(columns=self.field_names)
        self.flow_dict = {}
        self.origin = [32.9857, -96.7502]  # toyota NA HQ
        self.dest = [33.0811809, -96.841015]  # UTD
        self.val = None

    def create_csv_from_flow(self):
        # driver ranking: 1 worst, 4 best

        from datetime import datetime
        import calendar
        day = datetime.now().weekday()
        day_name = calendar.day_name[day]
        Tuesday = day_name == "Tuesday"
        Wednesday = day_name == "Wednesday"
        Thursday = day_name == "Thursday"
        Friday = day_name == "Friday"
        Saturday = day_name == "Saturday"
        Sunday = day_name == "Sunday"

        data = DataAPI().getFlow(None, None)
        abs_distance = wsg84_distance(self.origin, self.dest)
        weather = DataAPI().getWeatherAt([data[0]["location"]["shape"]["links"][0]["points"][0]["lat"],
                                          data[0]["location"]["shape"]["links"][0]["points"][0]["lng"]])
        for result in data:
            location = result["location"]
            if "description"  in location:
                name = location["description"]
            else:
                name = "Rd"
            print(name)
            Rd = "Rd" in name
            St = "St" in name
            Dr = "Dr" in name or "Ln" in name
            Ave = "Ave" in name or "Blvd" in name
            Route = "Route" in name
            Pike = "PKE" in name or "pke" in name
            Fwy = "Wy" in name or "wy" in name or "pky" in name
            flow = result["currentFlow"]
            su = flow["speedUncapped"]
            ff = flow["freeFlow"]
            flow_coeff = su / ff

            rain = False
            humidity = 0
            if "Rain" in weather["current"]["weather"]:
                rain = True
            humidity = float(weather["current"]["humidity"] )
            for i in location["shape"]["links"]:
                paths = i
                start = [paths["points"][0]["lat"], paths["points"][0]["lng"]]
                end = [paths["points"][-1]["lat"], paths["points"][-1]["lng"]]
                if (wsg84_distance(start, self.origin) + wsg84_distance(end, self.dest)) < 1.1 * abs_distance:
                    if tuple(start) not in self.flow_dict and tuple(end) not in self.flow_dict:
                        self.flow_dict[tuple(start)] = flow_coeff
                        self.flow_dict[tuple(end)] = flow_coeff
                        distance = paths["length"] * 0.000621371
                        duration = distance / ff * su * 100
                        self.df.loc[len(self.df.index)] = [int(Tuesday), int(Wednesday), int(Thursday), int(Friday), int(Saturday), int(Sunday),
                                                            duration, int(rain), int(Rd), int(St), int(Dr), int(Ave), int(Route),int(Pike) ,int(Fwy),
                                                           start[0], start[1], end[0], end[1], distance, humidity]
                        self.df = self.df.fillna(0)

                        print(len(self.df.index))
        if os.path.exists("street_data.csv"):
            os.remove("street_data.csv")
        self.df.to_csv("street_data.csv")

    def decompressFile(self, file):
        data = bz2.BZ2File(file, 'rb')
        data = pickle.load(data)
        return data

    def inference_data(self):
        f = open("street_data.csv")
        model = self.decompressFile('Model.pkl.pbz2')

        self.val     = model.predict(f)
        print(self.val)

    # def determine_avoid_bbox(self):
    #
    #         # some functions = determind the risk
    #         # score = func()
    #         for
    #         score *= (1- flow_coeff)
    #         if score >= driver_ranking:
    #                 temp_bbox = AvoidBoundingBox(max(start[0], end[0]),min(start[0], end[0]), max(start[1], end[1]), min(start[1], end[1]))
    #                 self.avoidance.append(temp_bbox)
    #
    #          return self.avoidance


to = toConnect()
to.inference_data()
