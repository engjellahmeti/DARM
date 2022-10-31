# @project Deviance Analysis by Means of Redescription Mining - Master Thesis 
# @author Engjëll Ahmeti
# @date 1/19/2021
from nlg.extraction import Extraction

class Negation:
    def __init__(self, rule, extract=True):
        self.rule = rule

        extraction = Extraction('not', self.rule, extract)

        self.tree = extraction.tree
        self.SPL = extraction.SPL
        self.CoNLL = extraction.CoNLL
        self.DSyntS = extraction.DSyntS

    def __str__(self):
        return self.rule.__str__()
