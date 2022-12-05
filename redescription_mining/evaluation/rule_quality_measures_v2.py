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


if not os.path.exists(r'redescription_mining\evaluation\rule_quality_measures_v2.json'):
    with open(r'redescription_mining\evaluation\rule_quality_measures_v2.json', 'w') as a:
        a.write('{}')

with open(r'redescription_mining\evaluation\rule_quality_measures_v2.json', 'r') as a:
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
        return 0.0

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

    return (1 / (total_amount_rules - 1.0)) * _jacc.sum() 

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


    return (1 / (total_amount_rules - 1.0)) * _jacc.sum()

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
def BF(support_activation, support_target, not_support_target):
    n_at = len(support_activation.intersection(support_target))
    n_atnot = len(support_activation) - n_at
    n_t = len(support_target)

    if n_atnot > 0 and n_t > 0:
        return (n_at * len(not_support_target))/((n_t * n_atnot) + .0)
    else:
        return 0

# Centered Confidence
def CenConf(support_activation, support_target, E):
    n_at = len(support_activation.intersection(support_target))
    n_a = len(support_activation)
    
    if E > 0 and n_a > 0:
        return ((E * n_at) - (n_a * len(support_target)))/((E * n_a) + .0)
    else:
        return 0

# Confidence
def Conf(support_activation, support_target):
    n_at = len(support_activation.intersection(support_target))
    n_a = len(support_activation)
    
    if n_a > 0:
        return n_at/(n_a + .0)
    else:
        return 0

# Information Gain
def IG(support_activation, support_target, E):
    n_at = len(support_activation.intersection(support_target))
    n_a = len(support_activation)
    n_t = len(support_target)

    if n_a > 0 and n_t > 0:
        if (E * n_at) /( n_a * n_t) > 0:
            return math.log((E * n_at) /( n_a * n_t))
    
    return 0

# Lift
def Lift(support_activation, support_target, E):
    n_at = len(support_activation.intersection(support_target))
    n_a = len(support_activation)
    n_t = len(support_target)

    if n_a > 0 and n_t > 0:
        return (E * n_at) /( n_a * n_t)
    else:
        return 0

# Pearson's correlation coefficient
def Pearson(support_activation, not_support_activation, support_target, not_support_target, E):
    n_at = len(support_activation.intersection(support_target))
    n_a = len(support_activation)
    n_t = len(support_target)
    n_anot = len(not_support_activation)
    n_tnot = len(not_support_target)

    if n_a > 0 and n_t > 0 and n_anot > 0 and n_tnot > 0 and E > 0:
        return (E * n_at - n_a * n_t) /math.sqrt(E * n_a * n_t * n_anot * n_tnot)
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
    lambda_get_rule_satisfactions = np.vectorize(get_rule_satisfactions)
    _jacc = lambda_get_rule_satisfactions(jacc.copy())

    bf_lambda = lambda x: BF(support_activation=rule_satisfied, support_target=x[0][1], not_support_target=x[0][2])
    lambda_bf_lambda = np.vectorize(bf_lambda)
    _jacc_bf = lambda_bf_lambda(_jacc.copy())
    bf = (1 / (total_amount_rules - 1.0)) * _jacc_bf.sum()

    cenconf_lambda = lambda x: CenConf(support_activation=rule_satisfied, support_target=x[0][1], E=E)
    lambda_cenconf_lambda = np.vectorize(cenconf_lambda)
    _jacc_cenconf= lambda_cenconf_lambda(_jacc.copy())
    cenconf = (1 / (total_amount_rules - 1.0)) * _jacc_cenconf.sum()

    conf_lambda = lambda x:  Conf(support_activation=rule_satisfied, support_target=x[0][1])
    lambda_conf_lambda = np.vectorize(conf_lambda)
    _jacc_conf = lambda_conf_lambda(_jacc.copy())
    conf = (1 / (total_amount_rules - 1.0)) * _jacc_conf.sum()

    ig_lambda = lambda x: IG(support_activation=rule_satisfied, support_target=x[0][1], E=E)
    lambda_ig_lambda = np.vectorize(ig_lambda)
    _jacc_ig = lambda_ig_lambda(_jacc.copy())
    ig = (1 / (total_amount_rules - 1.0)) * _jacc_ig.sum()
    
    lift_lambda = lambda x:  Lift(support_activation=rule_satisfied, support_target=x[0][1], E=E)
    lambda_lift_lambda = np.vectorize(lift_lambda)
    _jacc_lift = lambda_lift_lambda(_jacc.copy())
    lift = (1 / (total_amount_rules - 1.0)) * _jacc_lift.sum()
    
    pearson_lambda = lambda x: Pearson(support_activation=rule_satisfied, not_support_activation=rule_not_satisfied, support_target=x[0][1], not_support_target=x[0][2], E=E)
    lambda_pearson_lambda = np.vectorize(pearson_lambda)
    _jacc_pearson = lambda_pearson_lambda(_jacc.copy())
    pearson = (1 / (total_amount_rules - 1.0)) * _jacc_pearson.sum()

    return bf, cenconf, conf, ig, lift, pearson

def execute_redescription_quality_measures(metadata):
    algorithms = ['reremi', 'splittrees', 'new-approach']
    df = pd.DataFrame()
    if metadata['name'] not in rule_quality_measures.keys():
        rule_quality_measures[metadata['name']] = {}

    for algorithm in algorithms:
        if metadata['type'] not in rule_quality_measures[metadata['name']].keys():
            rule_quality_measures[metadata['name']][metadata['type']] = {}

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
            activation_view=r'feature_vectors\csv_feature_vectors\{0}\activation-{1}-{2}.csv'.format(metadata['type'], metadata['name'], str_representation),
            target_view=r'feature_vectors\csv_feature_vectors\{0}\target-{1}-{2}.csv'.format(metadata['type'], metadata['name'], str_representation),
            activation_attributes=None,
            target_attributes=None
        )

        temp = groups.get_group(dc)
        rules = pd.DataFrame(temp, columns=temp.columns)
        rules['declare_constraint'] = str_representation
        rules = rules[['algorithm', 'rid', 'query_activation','query_target', 'declare_constraint', 'activation_vars', 'target_vars']].to_dict()

        df_a, df_t, df_rules_satisfied = evaluate_rules_on_both_sides(redescription_data_model=rdm, rules=rules, for_deviant_traces=None)

        for position in rules['rid'].keys():
            _metadata = metadata.copy()
            _metadata['name'] = _metadata['name'] + '-' + rules['algorithm'][position]
            _ = support(df_a=df_a, df_t=df_t, rules=rules, position=position, metadata=_metadata, df_rules_satisfied=df_rules_satisfied)
    
    
        all_rules = {}
        for index, i in enumerate(rules['rid'].keys()):
            all_rules[index]= {
                'algorithm': rules['algorithm'][i],
                'rid': rules['rid'][i],
                'query_activation': rules['query_activation'][i],
                'query_target': rules['query_target'][i],
                'activation_vars': rules['activation_vars'][i],
                'target_vars': rules['target_vars'][i]

            }

        for i in all_rules.keys():
            temp = all_rules.copy()
            rule = temp[i]
            temp.pop(i, None)
            _aej = aej(rule=rule, position_of_rule=i, rules=temp, metadata=metadata)
            _aaj = aaj(rule=rule, position_of_rule=i, rules=temp)
            _r_size= r_size(rule)

            bf, cenconf, conf, ig, lift, pearson = define_values_for_measures(rule=rule, position_of_rule=i, rules=temp, metadata=metadata)
            
            if str_representation not in  rule_quality_measures[metadata['name']][metadata['type']].keys():
                rule_quality_measures[metadata['name']][metadata['type']][str_representation]= {}
            if rule['algorithm'] not in  rule_quality_measures[metadata['name']][metadata['type']][str_representation].keys():
                rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']] = {}
            if rule['rid'] not in rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']].keys():
                rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']][rule['rid']] = {}

            
            rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']][rule['rid']]['AEJ'] = _aej
            rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']][rule['rid']]['AAJ'] = _aaj
            rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']][rule['rid']]['R Size'] = _r_size
            rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']][rule['rid']]['Bayes Factor'] = bf
            rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']][rule['rid']]['Centered Confidence'] = cenconf
            rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']][rule['rid']]['Confidence'] = conf
            rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']][rule['rid']]['Information Gain'] = ig
            rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']][rule['rid']]['Lift'] = lift
            rule_quality_measures[metadata['name']][metadata['type']][str_representation][rule['algorithm']][rule['rid']]['Pearson correlation coefficient'] = pearson


        with open(r'redescription_mining\evaluation\rule_quality_measures_v2.json', 'w') as a:
            json.dump(rule_quality_measures, a)
