# @project Deviance Analysis by Means of Redescription Mining - Master Thesis
# @author Engjëll Ahmeti
# @date 12/10/2020


import os
import shutil
from clired import exec_clired
from clired.classData import Data
import pandas as pd
from pandas.core.frame import DataFrame
from log_print import Print
import re
from redescription_mining.data_model import RedescriptionDataModel

class RedescriptionMining:
    def __init__(self):
        self.configuration = ''

    def discover_redescriptions(self, redescription_data_model: RedescriptionDataModel, is_positive_or_negative_log: str, activation_activity: str, target_activity: str, algorithm: str = 'reremi', config_or_template='config', filename='results') -> DataFrame:
        Print.YELLOW.print('Started extrating redescriptions. ')
        if len(redescription_data_model.activation_attributes) == 0 and len(redescription_data_model.target_attributes) == 0:
            return DataFrame()

        if config_or_template == 'config':
            config_path ="redescription_mining/configs/config.txt"
        else:
            config_path ="redescription_mining/configs/template.txt"

        full_config_path = os.path.abspath(config_path)

        self.setConfiguration(full_config_path=full_config_path, algorithm=algorithm, redescription_data_model=redescription_data_model)

        exec_clired.run([None, full_config_path])

        self.reset_configuration(full_config_path)

        redescriptions = self.rename_redescriptions(redescriptions_path=os.path.abspath('__TMP_DIR__results.queries'), move_redescriptions_to_path=os.path.abspath('redescription_mining/results/') + '/' + filename + '-'+is_positive_or_negative_log+'.queries', redescription_data_model=redescription_data_model, activation_activity=activation_activity, target_activity=target_activity)

        Print.YELLOW.print('Redescriptions have been generated.')

        return redescriptions

    def setConfiguration(self, full_config_path: str, algorithm: str, redescription_data_model: RedescriptionDataModel):
        Print.YELLOW.print('Setting up configurations.')
        LHS_data = os.path.abspath(redescription_data_model.activation_view)
        RHS_data = os.path.abspath(redescription_data_model.target_view)

        xml_string = open(full_config_path, mode='r').read()
        if 'LHS_data.csv' not in xml_string:
            temp = re.sub('\.txt', '-sample.txt', full_config_path)
            xml_string = open(temp, mode='r').read()

        self.configuration = xml_string

        xml_string = xml_string.replace('LHS_data.csv', LHS_data)
        xml_string = xml_string.replace('RHS_data.csv', RHS_data)
        xml_string = xml_string.replace('algorithm.csv', algorithm)


        with open(full_config_path, mode='w') as a:
            a.write(xml_string)

    def reset_configuration(self, full_config_path: str):
        temp = re.sub('\.txt', '-sample.txt', full_config_path)
        xml_string = open(temp, mode='r').read()
        with open(full_config_path, mode='w') as a:
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
