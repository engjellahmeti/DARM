# @project Deviance Analysis by Means of Redescription Mining - Master Thesis 
# @author Engjëll Ahmeti
# @date 1/23/2021
from nlg.extraction import Extraction

class Disjunction:
    def __init__(self, leftSide, rightSide, extract=True):
        self.leftSide = leftSide
        self.rightSide = rightSide

        extraction = Extraction('or', leftSide, extract,rightSide)

        self.tree = extraction.tree
        self.SPL = extraction.SPL
        self.CoNLL = extraction.CoNLL
        self.DSyntS = extraction.DSyntS

    def __str__(self) -> str:
        return self.leftSide.__str__() + 'or ' + self.rightSide.__str__()