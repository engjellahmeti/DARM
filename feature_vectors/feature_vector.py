# @project Deviance Analysis by Means of Redescription Mining - Master Thesis

# @author Volodymyr Leno, Marlon Dumas, Fabrizio Maria Maggi, Marcello La Rosa,Artem Polyvyanyy
# @author Engjëll Ahmeti, updated as per need of the project
# @date 12/8/2020
from log_print import Print
from typing import Dict, Any

class FeatureVector:
    def __init__(self, fromm: Dict[str, Any] = None, to: Dict[str, Any] = None, remove_some_attributes: bool = False, trace: str = ''):
        if type(fromm) == dict:
            self.fromm = fromm
            self.to = to
        else:
            self.fromm = fromm.payload
            self.to = to.payload

        self.trace = trace

        if remove_some_attributes:
            # if ones want to remove the resource
            fromm_copy = {}
            for attribute in self.fromm.keys():
                if "lifecycle:transition" not in attribute:
                    fromm_copy[attribute] = self.fromm[attribute]
            self.fromm = fromm_copy

            to_copy = {}
            for attribute in self.to.keys():
                if "lifecycle:transition" not in attribute:
                    to_copy[attribute] = self.to[attribute]
            self.to = to_copy

    def __str__(self):
        return Print.BLUE.print(str(self.fromm) + " => " + str(self.to) + "")

    def __eq__(self, other):
        if type(other) == FeatureVector:
            return self.fromm == other.fromm and self.to == other.to
        return False

    def __hash__(self):
        return hash(self.fromm, self.to)
