# @project Deviance Analysis by Means of Redescription Mining - Master Thesis 
# @author Engjëll Ahmeti
# @date 10/23/2021
from typing import List


class RedescriptionDataModel:
    def __init__(self, activation_view: str, activation_attributes: List[str], target_view: str, target_attributes: List[str]):
        self.activation_view = activation_view
        self.activation_attributes = activation_attributes
        self.target_view = target_view
        self.target_attributes = target_attributes
