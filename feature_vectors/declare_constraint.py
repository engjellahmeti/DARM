# @project Deviance Analysis by Means of Redescription Mining - Master Thesis

# @author Volodymyr Leno, Marlon Dumas, Fabrizio Maria Maggi, Marcello La Rosa,Artem Polyvyanyy
# @author Engjëll Ahmeti, updated as per need of the project
# @date 12/8/2020
from log_print import Print


class DeclareConstraint:
    def __init__(self, rule_type: str, activation: str, target: str, activation_resource: str = None,
                 target_resource: str = None):
        if rule_type is not None and (activation_resource is not None or target_resource is not None):
            self.rule_type = rule_type.lower()
            self.activation = activation
            self.target = target

            self.activation_resource = activation_resource
            self.target_resource = target_resource

        elif rule_type is not None:
            self.rule_type = rule_type.lower()
            self.activation = activation
            self.target = target

    def __str__(self):
        if "recedence" in self.rule_type.lower():
            return Print.RED.print(self.rule_type + "(" + self.target + ", " + self.activation + ")")
        else:
            return Print.RED.print(self.rule_type + "(" + self.activation + ", " + self.target + ")")

    def __eq__(self, other):
        return self.activation == other.activation and self.target == other.target and self.rule_type == other.rule_type
