"""
    @Author: Engjëll Ahmeti
    @Date: 24.11.2022
    @LastUpdate: 25.11.2022
"""

from redescription_mining.evaluation.metrics import jaccard_index, support, evaluate_rules_on_both_sides
from redescription_mining.data_model import RedescriptionDataModel
import numpy as np
import json
import os
import pandas as pd
import re
import math


if not os.path.exists(r'redescription_mining\evaluation\rule_quality_measures.json'):
    with open(r'redescription_mining\evaluation\rule_quality_measures.json', 'w') as a:
        a.write('{}')

with open(r'redescription_mining\evaluation\rule_quality_measures.json', 'r') as a:
    rule_quality_measures = json.load(a)


def get_attributes_of_redescription(rule, attributes):
    att = []

    for attribute in attributes.split(','):
        attribute = str(attribute).strip() 
        if attribute in rule:
            att.append(attribute)
    
    return set(att)

def support_of_redescription(support_activation, support_target):
    return support_activation.intersection(support_target)

# The measure providing information about the redundancy of elements contained in the redescription support is called the average redescription element Jaccard index and is defined as
def aej(rule, position_of_rule, rules, metadata):
    total_amount_rules = len(rules) + 1
    jacc = np.arange(0, total_amount_rules)
    jacc = np.delete(jacc, position_of_rule)
    
    supp_act_r_i, supp_tar_r_i, _= support(df_a=None, df_t=None, rules=rule['rid'] + '?' + rule['algorithm'], position=0, metadata=metadata)

    if not supp_act_r_i:
        return

    supp_r_i = supp_act_r_i.intersection(supp_tar_r_i)

    # get supp(q1) & supp(q2) where r_j = (q1, q2)
    get_supports_of_descriptions = lambda x: [support(df_a=None, df_t=None, rules=rules[x]['rid'] + '?' + rules[x]['algorithm'], position=x, metadata=metadata)]

    # get sup(r_j) = supp(q1) intersection supp(q2) 
    apply_intersection_of_descriptions = lambda x: x[0][0].intersection(x[0][1])

    # apply jacard to rules
    apply_jaccard_to_redescriptions = lambda sup_r_j: jaccard_index(supp_r_i, sup_r_j)

    lambda_get_supports_of_descriptions = np.vectorize(get_supports_of_descriptions)
    lambda_apply_intersection_of_descriptions = np.vectorize(apply_intersection_of_descriptions)
    lambda_apply_jaccard_to_redescriptions = np.vectorize(apply_jaccard_to_redescriptions)


    _jacc = lambda_get_supports_of_descriptions(jacc.copy())
    _jacc = lambda_apply_intersection_of_descriptions(_jacc)
    _jacc = lambda_apply_jaccard_to_redescriptions(_jacc)


    return (1 / (total_amount_rules - 1)) * _jacc.sum()

#  the measure providing information about the redundancy of attributes contained in redescription queries, called the average redescription attribute Jaccard index, is defined as:
def aaj(rule, position_of_rule, rules):
    total_amount_rules = len(rules) + 1
    jacc = np.arange(0, total_amount_rules)
    jacc = np.delete(jacc, position_of_rule)
    
    attrs_r_i = get_attributes_of_redescription(rule=rule['query_activation'] + ' => ' + rule['query_target'], attributes=rule['activation_vars'] + ', ' + rule['target_vars'])


    # # get attrs(q1) & attrs(q2) where r_j = (q1, q2)
    get_attrs_of_redescriptions = lambda x: get_attributes_of_redescription(rule=rules[x]['query_activation'] + ' => ' + rules[x]['query_target'], attributes=rules[x]['activation_vars'] + ', ' + rules[x]['target_vars'])

    # # apply jacard to rules
    apply_jaccard_to_redescriptions = lambda attrs_r_j: jaccard_index(attrs_r_i, attrs_r_j)

    lambda_get_attrs_of_redescriptions = np.vectorize(get_attrs_of_redescriptions)
    lambda_apply_jaccard_to_redescriptions = np.vectorize(apply_jaccard_to_redescriptions)


    _jacc = lambda_get_attrs_of_redescriptions(jacc.copy())
    _jacc = lambda_apply_jaccard_to_redescriptions(_jacc)


    return (1 / (total_amount_rules - 1)) * _jacc.sum()

# To emphasize importance of the redescription size from the point of understandability (|a t t r(R)|≥20 considered to be highly complex to understand)
def r_size(rule):
    counter = 0
    _rule = rule['query_activation'] + ' => ' + rule['query_target']
    attributes = rule['activation_vars'] + ', ' + rule['target_vars']
    for attribute in attributes.split(','):
        _temp = re.findall(str(attribute).strip(), _rule, re.S|re.I)
        counter += len(_temp)
    
    return 1 if counter >= 20 else counter/20

# BayesFactor
def BF(n_a, n_b, n_notb):
    n_ab = len(n_a.intersection(n_b))
    n_abnot = len(n_a) - n_ab
    n_b = len(n_b)

    if n_abnot > 0 and n_b > 0:
        return (n_ab * len(n_notb))/((n_b * n_abnot) + .0)
    else:
        return 0

# Centerred Confidence
def CenConf(n_a, n_b, n):
    n_ab = len(n_a.intersection(n_b))
    n_a = len(n_a)
    
    if n > 0 and n_a > 0:
        return ((n * n_ab) - (n_a * len(n_b)))/((n * n_a) + .0)
    else:
        return 0

# Confidence
def Conf(n_a, n_b):
    n_ab = len(n_a.intersection(n_b))
    n_a = len(n_a)
    
    if n_a > 0:
        return n_ab/(n_a + .0)
    else:
        return 0

# Information Gain
def IG(n_a, n_b, n):
    n_ab = len(n_a.intersection(n_b))
    n_a = len(n_a)
    n_b = len(n_b)

    if n_a > 0 and n_b > 0:
        if (n * n_ab) /( n_a * n_b) != 0:
            return math.log((n * n_ab) /( n_a * n_b))
    
    return 0

# Lift
def Lift(n_a, n_b, n):
    n_ab = len(n_a.intersection(n_b))
    n_a = len(n_a)
    n_b = len(n_b)

    if n_a > 0 and n_b > 0:
        return (n * n_ab) /( n_a * n_b)
    else:
        return 0

# Pearson's correlation coefficient
def R(n_a, n_anot, n_b, n_bnot, n):
    n_ab = len(n_a.intersection(n_b))
    n_a = len(n_a)
    n_b = len(n_b)
    n_anot = len(n_anot)
    n_bnot = len(n_bnot)

    if n_a > 0 and n_b > 0 and n_anot > 0 and n_bnot > 0 and n > 0:
        return (n * n_ab - n_a * n_b) /math.sqrt(n * n_a * n_b * n_anot * n_bnot)
    else:
        return 0

# define values of n, n_a, n_b, n_ab & n_a!b from paper Association Rule Interestingness Measures: Experimental and TheoreticalStudies
def define_values_for_measures(rule, position_of_rule, rules, metadata):
    total_amount_rules = len(rules) + 1
    jacc = np.arange(0, total_amount_rules)
    jacc = np.delete(jacc, position_of_rule)
    
    _, _, _rule_satisfied = support(df_a=None, df_t=None, rules=rule['rid'] + '?' + rule['algorithm'], position=0, metadata=metadata)

    if _rule_satisfied[0] < 0:
        return

    E = _rule_satisfied[0]
    rule_satisfied = _rule_satisfied[1]
    rule_not_satisfied = _rule_satisfied[2]
    declare_constraint = _rule_satisfied[3]

    # get supp(q1) & supp(q2) where r_j = (q1, q2)
    get_rule_satisfactions = lambda x: [support(df_a=None, df_t=None, rules=rules[x]['rid'] + '?' + rules[x]['algorithm'], position=x, metadata=metadata)]

    # get sup(r_j) = supp(q1) intersection supp(q2) 
    apply_get_rules_from_the_same_declare_constraint = lambda r_j: r_j if declare_constraint == r_j[0][2][3] else None


    lambda_get_rule_satisfactions = np.vectorize(get_rule_satisfactions)
    lambda_apply_get_rules_from_the_same_declare_constraint = np.vectorize(apply_get_rules_from_the_same_declare_constraint)

    _jacc = lambda_get_rule_satisfactions(jacc.copy())
    _jacc = lambda_apply_get_rules_from_the_same_declare_constraint(_jacc)

    results = {'declare_constraint':declare_constraint , 'Bayes Factor': {}, 'Centered Confidence': {}, 'Confidence': {}, 'Information Gain': {}, 'Lift': {}, 'Pearson correlation coefficient': {}}
    for i, item in enumerate(_jacc):
        get_id = jacc[i]
        if item and str(item) != 'nan':
            results['Bayes Factor'][rules[get_id]['rid'] + '-' + rules[get_id]['algorithm']] = BF(n_a=rule_satisfied, n_b=item[0][1], n_notb=item[0][2])
            results['Centered Confidence'][rules[get_id]['rid'] + '-' + rules[get_id]['algorithm']] = CenConf(n_a=rule_satisfied, n_b=item[0][1], n=E)
            results['Confidence'][rules[get_id]['rid'] + '-' + rules[get_id]['algorithm']] = Conf(n_a=rule_satisfied, n_b=item[0][1])
            results['Information Gain'][rules[get_id]['rid'] + '-' + rules[get_id]['algorithm']] = IG(n_a=rule_satisfied, n_b=item[0][1], n=E)
            results['Lift'][rules[get_id]['rid'] + '-' + rules[get_id]['algorithm']] = Lift(n_a=rule_satisfied, n_b=item[0][1], n=E)
            results['Pearson correlation coefficient'][rules[get_id]['rid'] + '-' + rules[get_id]['algorithm']] = R(n_a=rule_satisfied, n_anot=rule_not_satisfied, n_b=item[0][1], n_bnot=item[0][2], n=E)

    return results

def execute_redescription_quality_measures(metadata):
    algorithms = ['reremi', 'splittrees', 'new-approach']
    df = pd.DataFrame()
    if metadata['name'] not in rule_quality_measures.keys():
        rule_quality_measures[metadata['name']] = {}

    for algorithm in algorithms:
        if algorithm not in rule_quality_measures[metadata['name']].keys():
            rule_quality_measures[metadata['name']][algorithm] = {}

        if not df.empty:
            df1 = pd.read_csv(r'redescription_mining\results\{0}-{1}-{2}.queries'.format(metadata['name'], algorithm, metadata['type']))
            df1['algorithm'] = algorithm
            df = pd.concat([df, df1], axis=0,ignore_index=True)
        else:
            df = pd.read_csv(r'redescription_mining\results\{0}-{1}-{2}.queries'.format(metadata['name'], algorithm, metadata['type']))
            df['algorithm'] = algorithm
    
    groups = df.groupby(['activation_activity', 'target_activity', 'constraint'])

    for dc in groups.groups:
        if "recedence" in dc[2]:
            str_representation = dc[2] + "(" + dc[1] + ", " + dc[0] + ")"
        else:
            str_representation = dc[2] + "(" + dc[0] + ", " + dc[1] + ")"
        
        rdm = RedescriptionDataModel(
            activation_view=r'feature_vectors\csv_feature_vectors\{0}\activation-{1}.csv'.format(metadata['type'], str_representation),
            target_view=r'feature_vectors\csv_feature_vectors\{0}\target-{1}.csv'.format(metadata['type'], str_representation),
            activation_attributes=None,
            target_attributes=None
        )

        temp = groups.get_group(dc)
        rules = pd.DataFrame(temp, columns=temp.columns)
        rules['declare_constraint'] = str_representation
        rules = rules[['algorithm', 'rid', 'query_activation','query_target', 'declare_constraint']].to_dict()

        df_a, df_t, df_rules_satisfied = evaluate_rules_on_both_sides(redescription_data_model=rdm, rules=rules, for_deviant_traces=None)

        for position in rules['rid'].keys():
            _metadata = metadata.copy()
            _metadata['name'] = _metadata['name'] + '-' + rules['algorithm'][position]
            _ = support(df_a=df_a, df_t=df_t, rules=rules, position=position, metadata=_metadata, df_rules_satisfied=df_rules_satisfied)
    
    
    all_rules_temp = df[['algorithm', 'rid', 'query_activation','query_target', 'activation_vars', 'target_vars']].to_dict()

    all_rules = {}
    for i in all_rules_temp['rid'].keys():
        all_rules[i]= {
            'algorithm': all_rules_temp['algorithm'][i],
            'rid': all_rules_temp['rid'][i],
            'query_activation': all_rules_temp['query_activation'][i],
            'query_target': all_rules_temp['query_target'][i],
            'activation_vars': all_rules_temp['activation_vars'][i],
            'target_vars': all_rules_temp['target_vars'][i]

        }

    for i in all_rules.keys():
        temp = all_rules.copy()
        rule = temp[i]
        temp.pop(i, None)
        _aej = aej(rule=rule, position_of_rule=i, rules=temp, metadata=metadata)
        _aaj = aaj(rule=rule, position_of_rule=i, rules=temp)
        _r_size= r_size(rule)

        # get the other metrics
        results = define_values_for_measures(rule=rule, position_of_rule=i, rules=temp, metadata=metadata)
        
        if rule['rid'] not in rule_quality_measures[metadata['name']][rule['algorithm']].keys():
            rule_quality_measures[metadata['name']][rule['algorithm']][rule['rid']] = {}


        rule_quality_measures[metadata['name']][rule['algorithm']][rule['rid']]['AEJ'] = _aej
        rule_quality_measures[metadata['name']][rule['algorithm']][rule['rid']]['AAJ'] = _aaj
        rule_quality_measures[metadata['name']][rule['algorithm']][rule['rid']]['R Size'] = _r_size
        rule_quality_measures[metadata['name']][rule['algorithm']][rule['rid']].update(results)    

    with open(r'redescription_mining\evaluation\rule_quality_measures.json', 'w') as a:
        json.dump(rule_quality_measures, a)
