# @project Deviance Analysis by Means of Redescription Mining - Master Thesis
# @author Engjëll Ahmeti
# @date 12/10/2020


import os
import shutil
from clired import exec_clired
import pandas as pd
from pandas.core.frame import DataFrame
from log_print import Print
import re
import time
from redescription_mining.data_model import RedescriptionDataModel
from redescription_mining.new_approach.discover_decision_trees import discover_the_best_tree
from redescription_mining.new_approach.transform_trees_into_redescription_rules import extract_rules, store_the_discovered_rules

class RedescriptionMining:
    def __init__(self):
        self.configuration = ''

    def discover_redescriptions(self, redescription_data_model: RedescriptionDataModel, is_positive_or_negative_log: str, activation_activity: str, target_activity: str, last_id: int, declare_constraint: str, algorithm: str = 'reremi', config_or_template='config', filename='results') -> DataFrame:
        Print.YELLOW.print('Started extrating redescriptions. ')
        if len(redescription_data_model.activation_attributes) == 0 and len(redescription_data_model.target_attributes) == 0:
            return DataFrame()

        path_to_store_discovered_redescription_rules = os.path.abspath('redescription_mining/results/') + '/' + filename + '-' + algorithm + '-'+is_positive_or_negative_log+'.queries'
        (lhs_len, rhs_len) = self.get_shapes_feature_vectors(redescription_data_model=redescription_data_model)
        if algorithm == 'new-approach':
            start_time_per_constraint = time.time()
            performance = discover_the_best_tree(negative_or_positive=is_positive_or_negative_log, activation_path=redescription_data_model.activation_view, target_path=redescription_data_model.target_view)
            end_time_per_constraint = round(time.time() - start_time_per_constraint, 2)

            redescriptions, last_id = extract_rules(performance=performance, negative_or_positive=is_positive_or_negative_log, last_id=last_id, declare_constraint=declare_constraint)

            redescriptions = store_the_discovered_rules(redescription_data_model=redescription_data_model, rules=redescriptions, metadata={'name': filename + '-' + algorithm, 'type':is_positive_or_negative_log}, activation_activity=activation_activity, target_activity=target_activity)
            
        else:
            if config_or_template == 'config':
                config_path ="redescription_mining/configs/config.txt"
            else:
                config_path ="redescription_mining/configs/template.txt"

            full_config_path = os.path.abspath(config_path)

            self.setConfiguration(full_config_path=full_config_path, redescription_data_model=redescription_data_model)

            start_time_per_constraint = time.time()
            exec_clired.run([None, full_config_path])
            end_time_per_constraint = round(time.time() - start_time_per_constraint, 2)

            self.reset_configuration(full_config_path)

            redescriptions = self.rename_redescriptions(redescriptions_path=os.path.abspath('__TMP_DIR__results.queries'), move_redescriptions_to_path=path_to_store_discovered_redescription_rules, redescription_data_model=redescription_data_model, activation_activity=activation_activity, target_activity=target_activity)

            Print.YELLOW.print('Redescriptions have been generated.')

        return redescriptions, lhs_len, rhs_len, end_time_per_constraint, last_id

    def get_shapes_feature_vectors(self, redescription_data_model: RedescriptionDataModel):
        return  pd.read_csv(redescription_data_model.activation_view, index_col=0).shape, pd.read_csv(redescription_data_model.target_view, index_col=0).shape

    def setConfiguration(self, full_config_path: str, redescription_data_model: RedescriptionDataModel):
        Print.YELLOW.print('Setting up configurations.')
        LHS_data = os.path.abspath(redescription_data_model.activation_view)
        RHS_data = os.path.abspath(redescription_data_model.target_view)

        xml_string = open(full_config_path, mode='r').read()
        if '@LHS_data' not in xml_string:
            temp = re.sub('\.txt', '-sample.txt', full_config_path)
            xml_string = open(temp, mode='r').read()

        self.configuration = xml_string

        xml_string = xml_string.replace('@LHS_data', redescription_data_model.activation_view)
        xml_string = xml_string.replace('@RHS_data', redescription_data_model.target_view)
        # xml_string = xml_string.replace('@algorithm', algorithm)

        with open(full_config_path, mode='w') as a:
            a.write(xml_string)

    def reset_configuration(self, full_config_path: str):
        temp = re.sub('\.txt', '-sample.txt', full_config_path)
        xml_string = open(temp, mode='r').read()
        with open(full_config_path, mode='w') as a:
            a.write(xml_string)
    
    def reset_all_configurations(self):
        full_config_path = os.path.abspath("redescription_mining/configs/config.txt")

        temp = re.sub('\.txt', '-sample-reset.txt', full_config_path)
        xml_string = open(temp, mode='r').read()
        with open(full_config_path, mode='w') as a:
            a.write(xml_string)

        with open(re.sub('\.txt', '-sample.txt', full_config_path), mode='w') as a:
            a.write(xml_string)

    def rename_redescriptions(self, redescriptions_path: str, move_redescriptions_to_path: str, redescription_data_model: RedescriptionDataModel, activation_activity: str, target_activity: str):
        redescriptions = pd.read_csv(redescriptions_path, delimiter='\t')
        l_vs = {}
        activation_vars = redescription_data_model.activation_attributes
        target_vars = redescription_data_model.target_attributes

        for i in range(0, len(activation_vars)):
            l_vs['v' +str(i)] = activation_vars[i]

        r_vs = {}
        for i in range(0, len(target_vars)):
            r_vs['v' +str(i)] = target_vars[i]

        activity_activation = []
        rules_activation = []
        activation_vars = []
        for row in redescriptions['query_LHS']:
            fields = ''
            activity_activation.append(activation_activity)
            for v in l_vs.keys():
                row = row.replace(v, l_vs[v])

                if l_vs[v] in row:
                    if fields == '':
                        fields = l_vs[v]
                    else:
                        fields = fields + ',' + l_vs[v]

            activation_vars.append(fields)
            rules_activation.append(row)

        redescriptions['query_LHS'] = rules_activation
        redescriptions['LHS_vars'] = activation_vars
        redescriptions['LHS_activity'] = activity_activation



        activity_target = []
        rules_target = []
        target_vars = []
        for row in redescriptions['query_RHS']:
            fields = ''
            activity_target.append(target_activity)
            for v in r_vs.keys():
                row = row.replace(v, r_vs[v])

                if r_vs[v] in row:
                    if fields == '':
                        fields = r_vs[v]
                    else:
                        fields = fields + ',' + r_vs[v]

            target_vars.append(fields)
            rules_target.append(row)

        redescriptions['query_RHS'] = rules_target
        redescriptions['RHS_vars'] = target_vars
        redescriptions['RHS_activity'] = activity_target

        redescriptions.drop(redescriptions.columns[9], axis=1, inplace=True)


        redescriptions.rename(columns={"query_LHS": "query_activation", "query_RHS": "query_target", "LHS_vars": "activation_vars", "LHS_activity": "activation_activity", "RHS_vars": "target_vars", "RHS_activity": "target_activity", "constraint": "constraint_type"}, inplace=True)

        redescriptions.to_csv(redescriptions_path, index=False)


        shutil.move(redescriptions_path, move_redescriptions_to_path)

        return redescriptions
