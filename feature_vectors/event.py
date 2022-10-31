# @project Deviance Analysis by Means of Redescription Mining - Master Thesis

# @author Volodymyr Leno, Marlon Dumas, Fabrizio Maria Maggi, Marcello La Rosa,Artem Polyvyanyy
# @author Engjëll Ahmeti, updated as per need of the project
# @date 12/8/2020

from log_print import Print


class Event:
    def __init__(self, values=None, caseID=None, activity_name=None, timestamp=None, event=None):
        if len(values) > 0:
            self.caseID = values[0]
            self.activity_name = values[1]
            self.timestamp = values[2]
            self.payload = values[3]

        elif caseID is not None:
            self.caseID = caseID
            self.activity_name = activity_name
            self.timestamp = timestamp
            self.payload = []

        elif event is not None:
            self.caseID = event.caseID
            self.activity_name = event.activity_name
            self.timestamp = event.timestamp
            self.payload = event.payload

    def __str__(self):
        return Print.BLUE.print(
            "(" + self.caseID + ", " + self.activity_name + ", " + self.timestamp + ", " + str(self.payload) + ")")
