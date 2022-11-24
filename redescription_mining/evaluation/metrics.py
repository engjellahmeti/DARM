from redescription_mining.evaluation.transform_redescription_to_boolean_value import rules_to_boolean_value
from redescription_mining.data_model import RedescriptionDataModel
import pandas as pd
import os
import math
import json
import numpy as np

if not os.path.exists('redescription_mining\evaluation\support.json'):
    with open('redescription_mining\evaluation\support.json', 'w') as a:
        a.write('{}')

with open('redescription_mining\evaluation\support.json', 'r') as a:
    supports = json.load(a)

def evaluate_rules_on_both_sides(redescription_data_model, rules):
    df_a, df_t = rules_to_boolean_value(df_a=pd.read_csv(redescription_data_model.activation_view, index_col=0), df_t=pd.read_csv(redescription_data_model.target_view, index_col=0), rules=rules)

    return df_a, df_t

def support(df_a, df_t, rules, position, metadata=None):
    does_not_exist = True
    if 'approach' not in metadata['name']:
        if metadata and metadata['name'] in supports.keys():
            if metadata['type'] in supports[metadata['name']].keys():
                if rules['rid'][position] in supports[metadata['name']][metadata['type']].keys():
                    true_activation = supports[metadata['name']][metadata['type']][rules['rid'][position]]['activation']
                    true_target = supports[metadata['name']][metadata['type']][rules['rid'][position]]['target']
                    does_not_exist = False

    if does_not_exist:
        true_activation = df_a.index[df_a[rules['rid'][position]]].tolist()
        true_target = df_t.index[df_t[rules['rid'][position]]].tolist()

        if metadata['name'] not in supports.keys():
            supports[metadata['name']] = {}
        
        if metadata['type'] not in supports[metadata['name']].keys():
            supports[metadata['name']][metadata['type']] = {}
        
        if rules['rid'][position] not in supports[metadata['name']][metadata['type']].keys():
            supports[metadata['name']][metadata['type']][rules['rid'][position]] = {}

        supports[metadata['name']][metadata['type']][rules['rid'][position]]['activation'] = true_activation
        supports[metadata['name']][metadata['type']][rules['rid'][position]]['target'] = true_target


        if 'approach' not in metadata['name']:
            with open('redescription_mining\evaluation\support.json', 'w') as a:
                json.dump(supports, a)
        

    return true_activation, true_target

def jaccard_index(supp_activation, supp_target):
    supp_activation = set(supp_activation)
    supp_target = set(supp_target)

    jaccard_index_ = len(supp_activation.intersection(supp_target))/(len(supp_activation.union(supp_target)) + .0)

    return jaccard_index_

def combination(E, n):
    prod_e = 1
    for i in range(E, 0, -1):
        prod_e *= i

    prod_n = 1
    for j in range(n, 0, -1):
        prod_n *= j
    
    prod_e_n = 1
    for k in range(E-n, 0, -1):
        prod_e_n *= k

    return prod_e/(prod_n * prod_e_n) 

def p_value(supp_activation, supp_target, E):
    p1 = len(supp_activation)/E
    p2 = len(supp_target)/E

    o = set(supp_activation).intersection(set(supp_target))

    n = np.arange(len(o), E + 1)
    
    lambda_pv = lambda _n: combination(E, _n) * math.pow((p1 * p2), _n) * math.pow((1- p1 * p2), (E-_n))
    lambda_pv = np.vectorize(lambda_pv)

    n = lambda_pv(n)
    
    return n.sum()
