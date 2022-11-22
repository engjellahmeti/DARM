"""
    @Author: Engjëll Ahmeti
    @Date: 22.11.2022
    @LastUpdate: 22.11.2022
"""

import re
from feature_vectors.declare_constraint import DeclareConstraint
from log_print import Print
from typing import Tuple, List

def get_declare_constraints(declare_file_path: str) -> List[DeclareConstraint]:
    Print.YELLOW.print("\nExtracting Declare Constraints")
    try:
        constraints = open(declare_file_path, 'rt').readlines()

        declare_constraints_temp: List[Tuple] = []

        for line in constraints:
            regex = re.search(r'([A-Za-z\s]*)\[([^,]*),\s*([^\]]*)\]\s*\|', line, re.S|re.I)
            if regex:
                dc = (str(regex.group(1)).lower(), str(regex.group(2)).strip(), str(regex.group(3)).strip())
                if dc not in declare_constraints_temp:
                    declare_constraints_temp.append(dc)

        declare_constraints: List[DeclareConstraint] = []

        for row in declare_constraints_temp:
            if "recedenc" in row[0]:
                declare_constraints.append(DeclareConstraint(rule_type=row[0], activation=row[2], target=row[1]))
            else:
                declare_constraints.append(DeclareConstraint(rule_type=row[0], activation=row[1], target=row[2]))


        Print.YELLOW.print("\nDeclare Constraints have been extracted.")

        return declare_constraints

    except Exception as e:
        Print.RED.print(e)