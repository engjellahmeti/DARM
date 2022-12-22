# @project Deviance Analysis by Means of Redescription Mining - Master Thesis
# @author Engjëll Ahmeti
# @date 12/8/2020

from pandas.core.frame import DataFrame
from feature_vectors.declare_constraint import DeclareConstraint
from feature_vectors.rule_extractor import RuleExtractor
from nlg.metrics import Metrics
from redescription_mining.redescription import RedescriptionMining
from rxes_approach.rxes_file import RXESApproach
from log_print import Print
from nlg.nlg import NLG
from redescription_mining.data_model import RedescriptionDataModel
from event_log_generation.get_declare_constraints import get_declare_constraints
from event_log_generation.generate_logs import generate_logs
from redescription_mining.evaluation.transform_redescription_to_boolean_value import rules_to_boolean_value
from mlflow import log_metrics, log_params, log_text, log_dict, set_tags, set_tag
from redescription_mining.evaluation.rule_quality_measures_v2 import execute_redescription_quality_measures
import os
import re
import pandas as pd
import sys, getopt
import random as rd
from typing import List, Optional, Tuple
import time
import json
import mlflow
import h2o

class Main:
    def __init__(self, extract_dsynts_on_leafs, algorithm, config_or_template, filename):
        self.algorithms = ['reremi', 'layeredtrees', 'splittrees', 'cartwheel']
        self.config_or_template = config_or_template
        self.filename = filename
        self.spl_trees = []
        self.metrics = Metrics()

        if config_or_template is not None:
            self.ruleExt = RuleExtractor()
            self.redesc = RedescriptionMining()
            self.rxes = RXESApproach()
            self.nlg_ = NLG(extract_dsynts_on_leafs=extract_dsynts_on_leafs)

            self.hyperparameters, self.relevant_hyperparameters = self.setup_hyperparameters()
            self.algorithm = self.hyperparameters['@mining_algo']

            with open('redescription_mining\execution_times.json', 'r') as a:
                self.execution_times = json.load(a)
                
    # region Helper Methods
    def discover_redescription_for_each_constraint(self, frame: DataFrame, event_log_path: str, declare_constraint: DeclareConstraint, is_positive_or_negative_log: str, filename: str, last_id: int) -> DataFrame:
        redescription_data_model: RedescriptionDataModel = self.ruleExt.extract_fulfilment(event_log_path=event_log_path,
                                                                                        declare_constraint=declare_constraint,
                                                                                        is_positive_or_negative_log=is_positive_or_negative_log,
                                                                                        filename=filename,
                                                                                        write_to_CSV=True,
                                                                                        remove_attributes=True)


        redescriptions, lhs_len, rhs_len, end_time_per_constraint, last_id = self.redesc.discover_redescriptions(redescription_data_model=redescription_data_model, is_positive_or_negative_log=is_positive_or_negative_log, activation_activity=declare_constraint.activation, target_activity=declare_constraint.target,
                                                                config_or_template=self.config_or_template, filename=self.filename, algorithm=self.algorithm, last_id=last_id, declare_constraint=declare_constraint.str_representation()) # algorithm='reremi')

        no_of_discovered_redescriptions = 0
        if not redescriptions.empty:
            no_of_discovered_redescriptions = redescriptions.shape[0]
            if frame is None:
                redescriptions['constraint'] = [declare_constraint.rule_type for x in
                                                        range(0, len(redescriptions.index))]
                frame = redescriptions.copy()
            else:
                redescriptions['constraint'] = [declare_constraint.rule_type for x in range(0, len(redescriptions.index))]
                frame = pd.concat([frame, redescriptions.copy()])

        return frame, lhs_len, rhs_len, end_time_per_constraint, no_of_discovered_redescriptions, last_id
    
    def write_final_descriptions(self, negative: DataFrame, positive: DataFrame, filename: str) -> None:
        if negative is not None:
            negative.to_csv(os.path.abspath(
                'redescription_mining/results/') + '/' + filename + '-' + self.algorithm + '-negative.queries', index=False)
        
        if positive is not None:
            positive.to_csv(os.path.abspath(
                'redescription_mining/results/') + '/' + filename + '-' + self.algorithm + '-positive.queries', index=False)
    
    def discover_redescriptions(self, declare_constraints: List[DeclareConstraint], negative_event_log_path: str, positive_event_log_path: str, filename: str, negative_or_positive: str = None):
        negative = None
        positive = None

        if self.algorithm == 'new-approach':
            negative = pd.DataFrame(columns=['rid', 'query_activation', 'query_target', 'acc', 'pval', 'card_Exo', 'card_Eox', 'card_Exx', 'card_Eoo', 'activation_vars', 'activation_activity', 'target_vars', 'target_activity', 'constraint'])
            positive = pd.DataFrame(columns=['rid', 'query_activation', 'query_target', 'acc', 'pval', 'card_Exo', 'card_Eox', 'card_Exx', 'card_Eoo', 'activation_vars', 'activation_activity', 'target_vars', 'target_activity', 'constraint'])
            if os.path.exists('redescription_mining\evaluation\support.json'):
                with open('redescription_mining\evaluation\support.json', 'r') as a:
                    supports = json.load(a)
                if filename + '-' + self.algorithm in supports.keys():
                    if negative_or_positive:
                        supports[filename + '-' + self.algorithm][negative_or_positive] = {}
                    else:
                        supports[filename + '-' + self.algorithm] = {}
                        supports[filename + '-' + self.algorithm]['negative'] = {}
                        supports[filename + '-' + self.algorithm]['positive'] = {}
                with open('redescription_mining\evaluation\support.json', 'w') as a:
                    json.dump(supports, a)
        
            from redescription_mining.evaluation.metrics import supports
            supports


        exec_data = {}
        start_time = time.time()
        tot_time = 0.0
        last_id, last_id_neg, last_id_pos = 0, 0, 0
        if negative_or_positive:
            for dc in declare_constraints:
                print()
                dc.__str__()
                negative, lhs_len, rhs_len, end_time_per_constraint, no_of_discovered_redescriptions, last_id = self.discover_redescription_for_each_constraint(frame=negative, event_log_path=negative_event_log_path, declare_constraint=dc, is_positive_or_negative_log=negative_or_positive, filename=filename, last_id=last_id)
                exec_data[dc.str_representation()] = {}
                exec_data[dc.str_representation()][negative_or_positive] = {
                    'activation_view_rows': lhs_len[0],
                    'activation_view_attributes': lhs_len[1],
                    'target_view_rows': rhs_len[0],
                    'target_view_attirbutes': rhs_len[1],
                    'no_of_discovered_redescriptions': no_of_discovered_redescriptions,
                    'execution_time': str(end_time_per_constraint) + 's'
                }
                tot_time += end_time_per_constraint

        else:
            for dc in declare_constraints:
                dc.__str__()
                exec_data[dc.str_representation()] = {}
               
                negative, lhs_len, rhs_len, neg_end_time_per_constraint, no_of_discovered_redescriptions, last_id_neg = self.discover_redescription_for_each_constraint(frame=negative, event_log_path=negative_event_log_path, declare_constraint=dc, is_positive_or_negative_log='negative', filename=filename, last_id=last_id_neg)
                exec_data[dc.str_representation()]['negative'] = {
                    'activation_view_rows': lhs_len[0],
                    'activation_view_attributes': lhs_len[1],
                    'target_view_rows': rhs_len[0],
                    'target_view_attirbutes': rhs_len[1],
                    'no_of_discovered_redescriptions': no_of_discovered_redescriptions,
                    'execution_time': str(neg_end_time_per_constraint) + 's'
                }

                positive, lhs_len, rhs_len, end_time_per_constraint, no_of_discovered_redescriptions, last_id_pos = self.discover_redescription_for_each_constraint(frame=positive, event_log_path=positive_event_log_path, declare_constraint=dc, is_positive_or_negative_log='positive', filename=filename, last_id=last_id_pos)
                exec_data[dc.str_representation()]['positive'] = {
                    'activation_view_rows': lhs_len[0],
                    'activation_view_attributes': lhs_len[1],
                    'target_view_rows': rhs_len[0],
                    'target_view_attirbutes': rhs_len[1],
                    'no_of_discovered_redescriptions': no_of_discovered_redescriptions,
                    'execution_time': str(end_time_per_constraint) + 's'
                }
                temp = round(neg_end_time_per_constraint + end_time_per_constraint, 2)
                exec_data[dc.str_representation()]['execution_time_per_constraint'] = str(temp) + 's'
                tot_time = round(tot_time + temp, 2)

                print()
        


        end_time = str(round(time.time() - start_time, 2)) + 's'
        # print('Execution time for the {0} algorithm in {2} event logs is {1}. '.format(self.algorithm, end_time, filename))
        self.redesc.reset_all_configurations()

        if filename not in self.execution_times.keys():
            self.execution_times[filename] = {}
        
        if self.algorithm not in self.execution_times[filename].keys():
            self.execution_times[filename][self.algorithm] = {}

        self.execution_times[filename][self.algorithm] = {
            'no_of_declare_constraints': len(declare_constraints),
            'total_execution_time_without_preprocessing': str(tot_time) + 's',
            'total_execution_time': end_time,
            'info_per_declare_constraint': exec_data
        }

        with open('redescription_mining\execution_times.json', 'w') as a:
            json.dump(self.execution_times, a)


        if negative_or_positive == 'positive':
            self.write_final_descriptions(negative=None, positive=negative, filename=filename)
        else:
            self.write_final_descriptions(negative=negative, positive=positive, filename=filename)

        return negative, positive

    def setup_hyperparameters(self):
        relevant_hyperparameters = {}
        with open('redescription_mining\hyperparameter.json', 'r') as a:
            hyperparameters = json.load(a)

        parameter_path = r'redescription_mining\configs\config.txt'
        xml_string = open(parameter_path, mode='r').read()
        
        relevant_found = False
        for parameter in hyperparameters.keys():
            xml_string = xml_string.replace(parameter, str(hyperparameters[parameter]))

            if not relevant_found:
                if parameter != '@inits_productivity':
                    relevant_hyperparameters[parameter.replace('@', '')] = hyperparameters[parameter]
                else:
                    relevant_found = True    

        with open(parameter_path, mode='w') as a:
            a.write(xml_string)

        parameter_path = parameter_path.replace('.txt', '-sample.txt')
        with open(parameter_path, mode='w') as a:
            a.write(xml_string)

        return hyperparameters, relevant_hyperparameters
    
    def reset_hyperparameters(self):
        xml_string = open(r'redescription_mining\configs\config-sample-reset.txt', mode='r').read()
        
        with open(r'redescription_mining\configs\config.txt', mode='w') as a:
            a.write(xml_string)

        with open(r'redescription_mining\configs\config-sample.txt', mode='w') as a:
            a.write(xml_string)
    
    def write_mlflow_logs1(self):
        # Log hyperparameters (key-value pair)
        log_params(self.relevant_hyperparameters)
        
        metrics = {}
        artifacts = {}

        for query in ['negative', 'positive']:
            path =  os.path.abspath('redescription_mining/results/') + '/{0}-{1}-{2}.queries'.format(self.filename, self.algorithm, query)
            artifacts['{0} {1} {2}'.format(self.filename, self.algorithm, query)] = path

            data = pd.read_csv(path)


            for rule in data.iterrows():
                rule = rule[1]
                constraint = rule['constraint']

                if 'prec' in constraint.lower():
                    met_constraint = '{0}_{2} _ {1}_'.format(constraint, rule['activation_activity'], rule['target_activity']) 
                    constraint = '{0}({2}, {1})'.format(constraint, rule['activation_activity'], rule['target_activity']) 

                else:
                    met_constraint = '{0}_{1} _ {2}_'.format(constraint, rule['activation_activity'], rule['target_activity']) 
                    constraint = '{0}({1}, {2})'.format(constraint, rule['activation_activity'], rule['target_activity']) 
                
                metrics[met_constraint + '-' + query + ' accuracy'] =  rule['acc']
                metrics[met_constraint + '-' + query + ' p-value'] =  rule['pval']

                if constraint + '-' + query not in artifacts.keys():
                    artifacts[constraint + '-' + query] = ''

                artifacts[constraint + '-' + query] = artifacts[constraint + '-' + query] + ', {0} ~ {1}'.format(rule['query_activation'], rule['query_target']) 


        log_dict(artifacts, 'artifact')
        log_metrics(metrics)
        set_tag('Algorithm', self.algorithm)

    def write_mlflow_logs(self):
        with open(r'redescription_mining\evaluation\rule_quality_measures_v2.json', 'r') as a:
            rule_quality_measures = json.load(a)

        if not os.path.exists('mlruns/'):
            os.makedirs('mlruns')
        
        mlflow_data = os.listdir('mlruns/')
        if '.trash' in mlflow_data:
            mlflow_data.remove('.trash')

        if len(mlflow_data) == 0:
            run_id = 1
        else:
            run_id = int(len(mlflow_data)/2) + 1

        for query in ['negative', 'positive']:
            temp_run_id = str(run_id) + '-' + ('ReReMi' if self.algorithm == 'reremi' else 'SplitT' if self.algorithm == 'splittrees' else 'New Approach') + '-' + query[0:3] + '-' + self.filename 
            experiment_id = mlflow.create_experiment(str(temp_run_id))
            path = 'redescription_mining/results/{0}-{1}-{2}.queries'.format(self.filename, self.algorithm, query)

            data = pd.read_csv(os.path.abspath(path))
            groups = data.groupby(by=['activation_activity', 'target_activity', 'constraint'])

            for name, group in groups:
                if 'prec' in name[2].lower():
                    constraint = '{0}({2}, {1})'.format(name[2], name[0], name[1]) 

                else:
                    constraint = '{0}({1}, {2})'.format(name[2], name[0], name[1]) 

                tags = {}
                tags['Declare Constraint'] = constraint
                groupOfRules = pd.DataFrame(group, columns=group.columns)
                for rule in groupOfRules.iterrows():
                    rule = rule[1]
                    rid = rule['rid']

                    metrics = rule_quality_measures[self.filename][query][constraint][self.algorithm][rid]
                    self.relevant_hyperparameters['Declare constraint'] = constraint
                    # metrics['accuracy'] =  rule['acc']
                    # metrics['p-value'] =  rule['pval']

                    tags['Redescription'] = '{0} ~ {1}'.format(rule['query_activation'], rule['query_target']) 

                    artifacts = {'results': path}
                    
                    with mlflow.start_run(run_name=query, experiment_id=experiment_id) as _:
                        log_metrics(metrics=metrics)
                        log_params(params=self.relevant_hyperparameters)
                        log_text('result', path)
                        log_dict(artifacts, 'artifact')
                        set_tags(tags)
                    
    # endregion

    # region 1. Input Declare File
    def input_declare_file(self, filename: str, declare_file_path: str, _generate_logs: bool = True, only_negative_logs: bool = False, amount_of_traces: int = 10000) -> Optional[Tuple[DataFrame, DataFrame]]:
        declare_constraints = get_declare_constraints(declare_file_path=declare_file_path)

        negative_event_log_path = os.path.abspath('event_log_reader/logs/' + filename + '-negative.xes')
        positive_event_log_path = os.path.abspath('event_log_reader/logs/' + filename + '-positive.xes')

        if only_negative_logs:
            generate_logs(declare_file_path=declare_file_path, event_log_location=negative_event_log_path)#, amount_of_traces=amount_of_traces)
               
        elif _generate_logs:
            generate_logs(declare_file_path=declare_file_path, event_log_location=positive_event_log_path, both_positive_negative_event=True, amount_of_traces=amount_of_traces)

        negative, positive = self.discover_redescriptions(declare_constraints=declare_constraints, negative_event_log_path=negative_event_log_path, positive_event_log_path=positive_event_log_path, filename=filename, negative_or_positive=None if not only_negative_logs else 'negative')

        Print.YELLOW.print('1. Input Declare File finished.')

        return (negative, positive)
    # endregion

    # region 2. Input real positive and negative event logs
    def input_real_positive_and_negative_event_logs(self, filename: str, positive_event_log_path: str, negative_event_log_path: str) -> Optional[Tuple[DataFrame, DataFrame]]:
        log_id = self.rxes.rxes(file_path=positive_event_log_path)

        declare_constraints = self.rxes.mine_constraints(filename=filename, log_id=2, no_of_rows=20)
        
        negative, positive = self.discover_redescriptions(declare_constraints=declare_constraints, negative_event_log_path=negative_event_log_path, positive_event_log_path=positive_event_log_path, filename=filename)

        Print.YELLOW.print('2. Input real positive and negative event logs finished.')

        return (negative, positive)
    #endregion

    # region 3. Input positive and negative event logs together with Declare Constraints
    def input_positive_and_event_logs_together_with_declare_constraints(self, positive_event_log_path: str, negative_event_log_path: str, declare_constraints: List[DeclareConstraint], filename) -> Optional[Tuple[DataFrame, DataFrame]]:
        negative, positive = self.discover_redescriptions(declare_constraints=declare_constraints, negative_event_log_path=negative_event_log_path, positive_event_log_path=positive_event_log_path, filename=filename)
        
        Print.YELLOW.print('3. Input positive and negative event logs together with Declare Constraints finished. ')

        return (negative, positive)
    #endregion

    # region 4. Input Declare File For One Type Only
    def input_declare_file_with_only_one_event_log(self, filename: str, is_positive_or_negative_log: str, declare_file_path: str, amount_of_traces: int = 1000) -> Optional[DataFrame]:
        declare_constraints = get_declare_constraints(declare_file_path=declare_file_path)
        event_log_path = os.path.abspath('event_log_reader/logs/' + filename + '-'+is_positive_or_negative_log+'.xes')

        generate_logs(declare_file_path=declare_file_path, event_log_location=event_log_path, amount_of_traces=amount_of_traces)
        
        output, _ = self.discover_redescriptions(declare_constraints=declare_constraints, negative_event_log_path=negative_event_log_path, positive_event_log_path=positive_event_log_path, filename=filename, negative_or_positive=is_positive_or_negative_log)
             
        Print.YELLOW.print('4. Input Declare File only one event log type finished.')

        return output

    # endregion
    
    # region X. NLG processing
    def nlg_call(self, negative_redescriptions_path, positive_redescriptions_path, print_bool=False, output=''):
        (set_of_rules, redescriptions) = self.nlg_.nlg(negative_redescriptions_path, positive_redescriptions_path)

        if print_bool and output == '':
            for item in set_of_rules:
                Print.CYAN.print('--> A negative event log is: ')
                Print.CYAN.print('            -->' + item[0].__str__())
                output += '--> A negative event log is:\n'
                output += '            --> {0}\n'.format(item[0].__str__())

                print()
                Print.CYAN.print('--> While a positive event log is: ')
                output += '\n'
                output += '--> While a positive event log is:\n'
                for itemP in item[1]:
                    Print.CYAN.print('            -->' + itemP.__str__())
                    output += '            --> {0}\n'.format(itemP.__str__())

        return (set_of_rules, redescriptions, output)

    def extract_spl_trees(self, tree):
        if type(tree) is list:
            for row in tree:
                for key in row.keys():
                    if key not in ['imply', 'or', 'and', 'not', 'parentheses']:
                        temp = row.copy()
                        self.spl_trees.append(temp[key]['spl'])
                    else:
                        self.extract_spl_trees(row[key])

        else:
            for key in tree.keys():
                if key not in ['imply', 'or', 'and', 'not', 'parentheses']:
                    temp = tree.copy()
                    self.spl_trees.append(temp[key]['spl'])
                else:
                    self.extract_spl_trees(tree[key])

    def save_spl_trees(self, negative_rules, positive_rules):
        path = os.path.abspath('nlg/spl_trees/' + self.filename + '-' + self.algorithm + '.spl')
        
        with open(path, 'wt') as a:
            a.write('--------------------Negative rules--------------------\n')
            a.write('\n')

            for negative_rule in negative_rules:
                a.write('   Rule {0}:  {1}\n'.format(negative_rule[1], re.sub(r'\.0', '', negative_rule[2])))

                self.extract_spl_trees(negative_rule[0].SPL)
                for spl in self.spl_trees:
                    a.write('       {0}\n'.format(spl.lower()))
                
                a.write('\n')

                self.spl_trees = []


            a.write('--------------------Positive rules--------------------\n')
            a.write('\n')

            for positive_rule in positive_rules:
                a.write('   Rule {0}:  {1}\n'.format(positive_rule[1], re.sub(r'\.0', '', positive_rule[2])))

                self.extract_spl_trees(positive_rule[0].SPL)
                for spl in self.spl_trees:
                    a.write('       {0}\n'.format(spl.lower()))

                a.write('\n')
                self.spl_trees = []
        
    def nlgCall_v2(self, negative_redescriptions_path, positive_redescriptions_path, print_bool=False):
        (negative_rules, positive_rules) = self.nlg_.nlgSplit(negative_redescriptions_path, positive_redescriptions_path)
        self.save_spl_trees(negative_rules=negative_rules, positive_rules=positive_rules)

        output = ''
        if print_bool:
            Print.CYAN.print('--> The negative event log has the following rules: ')
            output += '--> The negative event log has the following rules:\n'
            for index, item in enumerate(negative_rules):
                Print.END.print('            {0}. {1} ({2})'.format(index+1, item[0].__str__(), item[1]))
                output += '            {0}. {1} ({2})\n'.format(index+1, item[0].__str__(), item[1])

            print()

            Print.CYAN.print('--> While the positive event log has the following rules: ')
            output += '\n'
            output += '--> While the positive event log has the following rules:\n'
            for index, item in enumerate(positive_rules):
                Print.END.print('            {0}. {1} ({2})'.format(index+1, item[0].__str__(), item[1]))
                output += '            {0}. {1} ({2})\n'.format(index+1, item[0].__str__(), item[1])

        (setOfRules, redescriptions) = self.nlg_.nlg(negative_redescriptions_path, positive_redescriptions_path)
        output += '\n'
        return (setOfRules, redescriptions, output)

    def get_dsynts(self, set_of_rules):
        return self.nlg_.transform_conll_to_dsynts(set_of_rules)

    def generate_traces_for_nlg_example_output_created_new(self, declare_filename, filename, algorithm='reremi'):
        result = {}

        redescription_path = os.path.abspath('redescription_mining/results/' + filename + '-' + algorithm + '-positive.queries')
        if os.path.exists(redescription_path):
            rules = pd.read_csv(redescription_path).groupby(['activation_activity', 'target_activity', 'constraint'])

            negative_event_log_path = 'event_log_reader/logs/' + filename + '-negative.xes'

            declare_file_path = os.path.abspath('event_log_generation/declare constraint files/{0}.decl'.format(declare_filename))
            declare_constraints = get_declare_constraints(declare_file_path=declare_file_path)

            for dc in declare_constraints:
                is_positive_or_negative_log = 'negative'
                redescription_data_model: RedescriptionDataModel = self.ruleExt.extract_fulfilment(event_log_path=negative_event_log_path,
                                                                declare_constraint=dc,
                                                                is_positive_or_negative_log=is_positive_or_negative_log,
                                                                write_to_CSV=True,
                                                                remove_attributes=True, filename=filename)
                

                temp_group = rules.get_group((dc.activation, dc.target, dc.rule_type))
                dc_rules = pd.DataFrame(temp_group, columns=temp_group.columns)
                dc_rules = dc_rules[['rid', 'query_activation','query_target']].to_dict()

                output = rules_to_boolean_value(df_a=pd.read_csv(redescription_data_model.activation_view, index_col=0), df_t=pd.read_csv(redescription_data_model.target_view, index_col=0), rules=dc_rules, for_deviant_traces=True)

                _rules = {}
                for column in output.columns:
                    _rules[column] = output[column].where(output[column] == False)
               
                for key in output.keys():
                    if key not in result.keys():
                        result[key] = output[key]
                    else:
                        for rule in output[key].keys():
                            if rule not in result[key].keys():
                                result[key][rule] = output[key][rule]
                            else:
                                result[key][rule] = list(set(result[key][rule] + output[key][rule]))



         
        return result

    def generate_traces_for_nlg_example_output(self, declare_filename, filename, algorithm='reremi'):
        negative_event_log_path = 'event_log_reader/logs/' + filename + '-negative.xes'
        positive_event_log_path = 'event_log_reader/logs/' + filename + '-positive.xes'

        declare_file_path = os.path.abspath('event_log_generation/declare constraint files/{0}.decl'.format(declare_filename))
        declare_constraints = get_declare_constraints(declare_file_path=declare_file_path)

        result = {}
        for dc in declare_constraints:
            is_positive_or_negative_log = 'negative'
            _ = self.ruleExt.extract_fulfilment(event_log_path=negative_event_log_path,
                                                            declare_constraint=dc,
                                                            is_positive_or_negative_log=is_positive_or_negative_log,
                                                            write_to_CSV=True,
                                                            remove_attributes=True, filename=filename)


            if os.path.exists(os.path.abspath('redescription_mining/results/' + filename + '-' + algorithm + '-positive.queries')):
                output = self.nlg_.find_deviant_traces(filename + '-' + algorithm)

                if len(result.keys()) == 0:
                    result = output
                else:
                    for key in output.keys():
                        if key not in result.keys():
                            result[key] = output[key]
                        else:
                            for rule in output[key].keys():
                                if rule not in result[key].keys():
                                    result[key][rule] = output[key][rule]
                                else:
                                    result[key][rule] = list(set(result[key][rule] + output[key][rule]))



         
        return result

    def contraint_instance_extraction(self, is_positive_or_negative_log, declare_filename, filename, algorithm='reremi'):
        event_log_path = 'event_log_reader/logs/' + filename + '-{0}.xes'.format(is_positive_or_negative_log)

        declare_file_path = os.path.abspath(
            'event_log_generation/declare constraint files/{0}.decl'.format(declare_filename))
        declare_constraints = get_declare_constraints(declare_file_path=declare_file_path)

        for dc in declare_constraints:
            (_, _, _, _) = self.ruleExt.extract_fulfilment(event_log_path=event_log_path,
                                                            declare_constraint=dc,
                                                            is_positive_or_negative_log=is_positive_or_negative_log,
                                                            write_to_CSV=True,
                                                            remove_attributes=True)
            print()
        Print.YELLOW.print('Constraint Instance Extraction done. ')

    def print_trace_failure(self, traces, k):
        Print.GREEN.print('Concrete examples of traces that failed: ')
        output = 'Concrete examples of traces that failed:\n'

        if len(traces) > 0:
            for trace in traces:
                if len(traces[trace].keys()) >= 3:
                    rules_total = list(traces[trace].keys())
                    rules = [rd.choice(rules_total)]
                    temp = [traces[trace][rules[0]][0]]

                    rules_total.remove(rules[0])
                    rules.append(rd.choice(rules_total))
                    for item in traces[trace][rules[1]]:
                        if 'executed' in item:
                           temp.append(item)

                    if len(temp) == 1:
                        temp.append(traces[trace][rules[1]][0])

                    rules_total.remove(rules[1])
                    rules.append(rd.choice(rules_total))
                    temp.append(traces[trace][rules[2]][0])

                    Print.END.print('The process execution with \'{0}\' is deviant because {1}({2}), {3}({4}) and {5}({6}).'.format(Print.BLUE.__call__(trace), temp[0], rules[0], temp[1], rules[1], temp[2], rules[2]))
                    output += 'The process execution with \'{0}\' is deviant because {1}({2}), {3}({4}) and {5}({6}).\n'.format(trace, temp[0], rules[0], temp[1], rules[1], temp[2], rules[2])

                elif len(traces[trace].keys()) == 2:
                    rules = list(traces[trace].keys())

                    temp = None
                    for item in traces[trace][rules[0]]:
                        if 'executed' in item:
                            temp = [item]
                    if temp is None:
                        temp = [traces[trace][rules[0]][0]]
                    temp.append(traces[trace][rules[1]][0])
                    Print.END.print('The process execution with \'{0}\' is deviant because {1}({2}) and {3}({4}).'.format(Print.BLUE.__call__(trace),temp[0], rules[0], temp[1], rules[1]))
                    output += 'The process execution with \'{0}\' is deviant because {1}({2}) and {3}({4}).\n'.format(trace,temp[0], rules[0], temp[1], rules[1])

                else:
                    rules = list(traces[trace].keys())
                    # temp = traces[trace][rules[0]]
                    temp = None
                    for item in traces[trace][rules[0]]:
                        if 'executed' in item:
                            temp = [item]
                    if temp is None:
                        temp = [traces[trace][rules[0]]]
                    
                    if type(temp) is list and len(temp) > 0:
                        temp = temp[0]

                    Print.END.print('The process execution with \'{0}\' is deviant because {1}({2}).'.format(Print.BLUE.__call__(trace),temp[0], rules[0]))
                    output += 'The process execution with \'{0}\' is deviant because {1}({2}).\n'.format(trace,temp[0], rules[0])

        return output
    # endregion

    # region NLG Metrics
    def calculate_metrics(self, gen_file_path, ref_file_path, n_for_rouge):
        file_ref = open(ref_file_path, 'r')
        ref = file_ref.readlines()
        # ref = list(filter(('\n').__ne__, ref))
        ref.remove('--> The negative event log has the following rules:\n')
        ref.remove('--> While the positive event log has the following rules:\n')

        file_gen = open(gen_file_path, 'r')
        gen = file_gen.readlines()
        # gen = list(filter(('\n').__ne__, gen))
        gen.remove('--> The negative event log has the following rules:\n')
        gen.remove('--> While the positive event log has the following rules:\n')

        for i, l in enumerate(gen):
            gen[i] = re.sub(r'\n', '', l).strip()

        for i, l in enumerate(ref):
            ref[i] = re.sub(r'\n', '', l).strip()
        
        ter_score = self.metrics.ter(ref, gen)
        bleu_score = self.metrics.bleu(ref, gen)
        rouge_score = self.metrics.rouge_n(ref, gen, n=n_for_rouge)

        return (ter_score, bleu_score, rouge_score)

    def count_rules(self, filename, algorithm):
        deviant_df = pd.read_csv(os.path.abspath('redescription_mining/results/' + filename + '-' + algorithm + '-negative.queries'))
        positive_df = pd.read_csv(os.path.abspath('redescription_mining/results/' + filename + '-' + algorithm + '-positive.queries'))
        return (positive_df.shape[0], deviant_df.shape[0])

    def text_evaluation_metrics(self, filename=None, algorithm=None, n_for_rouge=2):
        results = {}
        if not filename or not algorithm:
            list_of_files = os.listdir('nlg/output/')
            for file in list_of_files:
                regex = re.search(r'(.*?)-(reremi|splittrees)', file, re.S|re.I)
                filename = re.sub('-', ' ', regex.group(1))
                algorithm = regex.group(2)
                
                if algorithm == 'splittrees':
                    if filename == 'credit application subset' or filename == 'running example':
                        # continue
                        pass
                
                count_positive_rules, count_deviant_rules = self.count_rules(regex.group(1), algorithm)
                gen_file_path = os.path.abspath('nlg/output/' + file)
                ref_file_path = os.path.abspath('nlg/target/' + file)
                (ter_score, bleu_score, rouge_score) = self.calculate_metrics(gen_file_path, ref_file_path, n_for_rouge)
                results[filename + ' ' + algorithm] = {
                        "filename": filename,
                        "algorithm": algorithm,
                        "BLEU (Bilingual Evaluation Understudy Score)": bleu_score, 
                        "ROUGE (Recall Oriented Understudy for Gisting Evaluation)": rouge_score, 
                        "TER (Translation Edit Rate)": ter_score,
                        "Count of Positive Rules": count_positive_rules,
                        "Count of Deviant Rules": count_deviant_rules                       
                }

            return results

        gen_file_path = os.path.abspath('nlg/output/' + filename + '-' + algorithm + '.txt')
        ref_file_path = os.path.abspath('nlg/target/'  + filename + '-' + algorithm + '.txt')
        count_positive_rules, count_deviant_rules = self.count_rules(filename, algorithm)

        (ter_score, bleu_score, rouge_score) = self.calculate_metrics(gen_file_path, ref_file_path, n_for_rouge)
        results[filename + ' ' + algorithm] = {
                        "filename": filename,
                        "algorithm": algorithm,
                        "BLEU (Bilingual Evaluation Understudy Score)": bleu_score, 
                        "ROUGE (Recall Oriented Understudy for Gisting Evaluation)": rouge_score, 
                        "TER (Translation Edit Rate)": ter_score,
                        "Count of Positive Rules": count_positive_rules,
                        "Count of Deviant Rules": count_deviant_rules   
                }

        return results

    #endregion

    # region Extract Arguments
    def extract_arguments(self, argv):
        input_type: int = -1
        algorithm: str = ''
        filename: str = ''
        extract_dsynts: bool = True 
        declare_filename: str = ''
        amount_of_traces: int = -1
        min_trace_length: int = -1
        max_trace_length: int = -1

        try:
            opts, args = getopt.getopt(argv, "h:t:a:f:e:d:s:i:x:",
                                       longopts=["input_type=", "algorithm=", "filename=", "extract_dsynts=",
                                                 "declare_filename=", "amount_of_traces=", "min_trace_length=",
                                                 "max_trace_length="])
        except getopt.GetoptError:
            print(getopt.GetoptError.msg)
            print(
                'python main.py -t <input_type> -a <algorithm> -f <filename> -e <extract_dsynts> -d <declare_filename> -s <amount_of_traces> -i <min_trace_length> -x <max_trace_length>')
            print(r"python main.py -t 1 -a 'reremi' -f 'test' -e True -d 'C:\...\test.decl' -s 1000 -i 2 -x 3")
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print(
                    'python main.py -t <input_type> -a <algorithm> -f <filename> -e <extract_dsynts> -d <declare_filename> -s <amount_of_traces> -i <min_trace_length> -x <max_trace_length>')
                print(r"python main.py -t 8 -a 'reremi' -f 'running-example' -e True -d 'Running Example.decl' -s 1000 -i 2 -x 3")
                sys.exit()
            elif opt in ("-t", "--input_type"):
                input_type = int(arg)
            elif opt in ("-a", "--algorithm"):
                algorithm = arg
            elif opt in ("-f", "--filename"):
                filename = arg
            elif opt in ("-e", "--extract_dsynts"):
                extract_dsynts = bool(arg)
            elif opt in ("-d", "--declare_filename"):
                declare_filename = arg
            elif opt in ("-s", "--amount_of_traces"):
                amount_of_traces = int(arg)
            elif opt in ("-i", "--min_trace_length"):
                min_trace_length = int(arg)
            elif opt in ("-x", "--max_trace_length"):
                max_trace_length = int(arg)

        print('{0}-{1}-{2}-{3}-{4}-{5}-{6}-{7}'.format(input_type, algorithm, filename, extract_dsynts, declare_filename,
                                                        amount_of_traces, min_trace_length, max_trace_length))
        return (input_type, algorithm, filename, extract_dsynts, declare_filename, amount_of_traces, min_trace_length, max_trace_length)
    
    # endregion

if __name__ == '__main__':
    # execute_redescription_quality_measures({'name': 'running-example', 'type':'negative'})
    Print.YELLOW.print('The tool has started. ')
    config_or_template = 'template' # 'config'
    main = Main(extract_dsynts_on_leafs=None, algorithm=None, config_or_template=None, filename=None)

    if len(sys.argv[1:]) > 0:
        (input_type, algorithm, filename, extract_dsynts_on_leafs, declare_filename, amount_of_traces, min_trace_length, max_trace_length) = main.extract_arguments(sys.argv[1:])
        main = Main(extract_dsynts_on_leafs=extract_dsynts_on_leafs, algorithm=algorithm, config_or_template=config_or_template, filename=filename)
        algorithm = main.algorithm

    else:
        extract_dsynts_on_leafs = False
        input_type = 3
        # algorithm = 'splittrees' # 'splittrees' reremi
        config_or_template = 'config' # 'config'
        filename = 'running-example'#credit-application-subset' #running-example' # road-traffic-fines,repair-example
        declare_filename = 'Running Example'#Credit Application Subset'#' # 'FirstPaperExample'  # Repair Example, Road Traffic Fines
        main = Main(extract_dsynts_on_leafs=extract_dsynts_on_leafs, algorithm=None, config_or_template=config_or_template, filename=filename)
        algorithm = main.algorithm
    
    if algorithm != 'reremi' and algorithm != 'splittrees':
        h2o.init()
        h2o.no_progress()

    if input_type == 1:
        _generate_logs = True
        only_negative_logs = False
        declare_file_path = os.path.abspath('event_log_generation/declare constraint files/{0}.decl'.format(declare_filename))

        negBool, posBool = True, True
        amount_of_traces = 1000
        while (negBool and posBool) or negBool:
            if posBool:
                (negative, positive) = main.input_declare_file(filename=filename, declare_file_path=declare_file_path, _generate_logs=_generate_logs, only_negative_logs=only_negative_logs, amount_of_traces=amount_of_traces)
                if positive is not None and negative is not None:
                    posBool = False
                    negBool = False
                elif positive is not None:
                    posBool = False
            else:
                break
                onlyNegative = True
                (negative, positive) = main.input_declare_file(filename=filename, declare_file_path=declare_file_path, _generate_logs=_generate_logs, only_negative_logs=only_negative_logs)
                if negative.shape[0] > 0:
                    negBool = False
                
            amount_of_traces += 1000
            if amount_of_traces > 16000:
                break

    elif input_type == 2:
        negative_event_log_path = 'event_log_reader/logs/'+filename+'-negative.xes'
        positive_event_log_path = 'event_log_reader/logs/'+filename+'-positive.xes'

        (negative, positive) = main.input_real_positive_and_negative_event_logs(filename=filename, positive_event_log_path=positive_event_log_path, negative_event_log_path=negative_event_log_path)

    elif input_type == 3:
        negative_event_log_path = 'event_log_reader/logs/'+filename+'-negative.xes'
        positive_event_log_path = 'event_log_reader/logs/'+filename+'-positive.xes'

        declare_file_path = os.path.abspath('event_log_generation/declare constraint files/{0}.decl'.format(declare_filename))

        declare_constraints = get_declare_constraints(declare_file_path=declare_file_path)

        (negative, positive) = main.input_positive_and_event_logs_together_with_declare_constraints(positive_event_log_path=positive_event_log_path, negative_event_log_path=negative_event_log_path, declare_constraints=declare_constraints, filename=filename)

    elif input_type == 4:
        declare_file_path = os.path.abspath('event_log_generation/declare constraint files/{0}.decl'.format(declare_filename))

        output = main.input_declare_file_with_only_one_event_log(is_positive_or_negative_log='positive', filename=filename, declare_file_path=declare_file_path)

    if input_type > 3:# or (negative is not None and positive is not None):
        traces = main.generate_traces_for_nlg_example_output(declare_filename=declare_filename, filename=filename, algorithm=algorithm)
        #
        negative_redescription_path = os.path.abspath('redescription_mining/results/'+filename+'-'+algorithm+'-negative.queries')
        positive_redescription_path = os.path.abspath('redescription_mining/results/'+filename+'-'+algorithm+'-positive.queries')
        (set_of_rules, redescriptions, output) = main.nlgCall_v2(negative_redescriptions_path=negative_redescription_path, positive_redescriptions_path=positive_redescription_path, print_bool=True)

        output_compare = main.nlg_.apply_comparisons(set_of_rules=set_of_rules, filename=filename +'-'+algorithm)
        
        output += output_compare
        # print()
        output_deviant = main.print_trace_failure(traces=traces, k=5)

        output += output_deviant

        output_path = os.path.abspath('nlg/output/'+filename+'-'+algorithm+'.txt')
        with open(output_path, 'wt') as a:
            a.write(output)
        
        
        results = main.text_evaluation_metrics(filename, algorithm) #os.path.abspath('nlg/output/repair-example-reremi.txt'))

        for key in results.keys():
            result = results[key]
            print('The metrics for event log "{0}" using {1} algorithm:'.format(result['filename'], result['algorithm']))
            for metric_key in result.keys():
                if metric_key != 'filename' and metric_key != 'algorithm':
                    print('      {0}: {1}'.format(metric_key, result[metric_key]))
            print()

    main.write_mlflow_logs()
    main.reset_hyperparameters()
    print()
