# @project Deviance Analysis by Means of Redescription Mining - Master Thesis 
# @author Engjëll Ahmeti
# @date 12/17/2020

import os
import re
from feature_vectors.declare_constraint import DeclareConstraint
import os
from log_print import Print
from typing import Tuple, List

class GenerateEventLogs:
    def generate_logs(self, declare_file_path: str, event_log_location: str, min_trace_length: int = 4, max_trace_length: int = 6, amount_of_traces: int = 1000, vacuity_enabled: str = '', even_length_distribution: str = '-eld', shuffle: str = '-shuffle 0', both_positive_negative_event: bool = False, max_same_instances: int = 1):
        path_alloy_generator = os.path.abspath('event_log_generation/tool/AlloyLogGenerator.jar')
        try:
            Print.YELLOW.print("\nGenerating Artificial Event Logs")
            if both_positive_negative_event:
                event_log_location_negative = re.sub('-positive.xes', '-negative.xes', event_log_location)
                os.system('java -jar "{0}" {1} {2} {3} "{4}" "{5}" {6} {7} {8} {9} {10}'.format(path_alloy_generator, min_trace_length, max_trace_length, amount_of_traces, declare_file_path, event_log_location, vacuity_enabled, '', even_length_distribution, shuffle, '-msi {0}'.format(max_same_instances)))
                os.system('java -jar "{0}" {1} {2} {3} "{4}" "{5}" {6} {7} {8} {9} {10}'.format(path_alloy_generator, min_trace_length, max_trace_length, amount_of_traces, declare_file_path, event_log_location_negative, vacuity_enabled, '-negative', even_length_distribution, shuffle, '-msi {0}'.format(max_same_instances)))


            else:
                if '-negative' in event_log_location:
                    os.system('java -jar "{0}" {1} {2} {3} "{4}" "{5}" {6} {7} {8} {9} {10}'.format(path_alloy_generator,
                                                                                               min_trace_length,
                                                                                               max_trace_length,
                                                                                               amount_of_traces,
                                                                                               declare_file_path,
                                                                                               event_log_location,
                                                                                               vacuity_enabled,
                                                                                               '-negative',
                                                                                               even_length_distribution,
                                                                                               shuffle, 
                                                                                               '-msi {0}'.format(max_same_instances)))
                else:
                    os.system('java -jar "{0}" {1} {2} {3} "{4}" "{5}" {6} {7} {8} {9} {10}'.format(path_alloy_generator,
                                                                                               min_trace_length,
                                                                                               max_trace_length,
                                                                                               amount_of_traces,
                                                                                               declare_file_path,
                                                                                               event_log_location,
                                                                                               vacuity_enabled,
                                                                                               '',
                                                                                               even_length_distribution,
                                                                                               shuffle, 
                                                                                               '-msi {0}'.format(max_same_instances)))

            os.remove(os.path.abspath('temp.als'))

            Print.YELLOW.print("\nArtificial Event Logs were generated.")

        except Exception as e:
            Print.RED.print(e)

    def get_declare_constraints(self, declare_file_path: str) -> List[DeclareConstraint]:
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