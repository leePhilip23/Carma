from data import DataAPI
from map import MapAPI
class toConnect():
        def __init__(self):
                self.avoidance = []
        def determine_avoidance(self,driver_ranking):
                # driver ranking: 1 worst, 4 best
                if driver_ranking == None:
                        driver_ranking = 2
                data = DataAPI().getFlow(None,None)
                for result in data:
                        location = result["location"]
                        name = location["description"]
                        flow = location["currentFlow"]
                        su = flow["speedUncapped"]
                        ff = flow["freeFlow"]
                        flow_coeff = su/ff
                        for i in location["shape"]["links"]:
                                score = 0
                                paths = i
                                start = [paths["points"][0]["lat"],paths["points"][0]["lng"]]
                                end = [paths["points"][-1]["lat"],paths["points"][-1]["lng"]]
                                distance = paths["length"]
                                # some functions = determind the risk
                                # score = func()
                                score *= flow_coeff
                                if score >= driver_ranking:
                                        self.avoidance.append(f"st")


toConnect().find_place_to_avoid()