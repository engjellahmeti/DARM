"""
    @Author: Engjëll Ahmeti
    @Date: 20.11.2022
    @LastUpdate: 21.11.2022
"""

import json
import jmespath
import re
import h2o
import os
from redescription_mining.new_approach.binary_tree import BinaryTree
from redescription_mining.new_approach.discover_decision_trees import discover_the_best_tree
from redescription_mining.evaluation.metrics import evaluate_rules_on_both_sides, support, jaccard_index, p_value
from redescription_mining.data_model import RedescriptionDataModel
import pandas as pd

def change_label_name(label, keys):
    if ('<' not in label and '>' not in label) and label in keys:
        keys = sorted(keys)
        index = keys.index(label)

        if index + 1 < len(keys):
            if keys[index+1] != label:
                label = label + '-1'
        elif index + 1 == len(keys):
            label = label + '-1'
        else:
            for i in range(index+1, len(keys)):
                node_lab = keys[i]
                if label in node_lab:
                    re_search = re.search(r'{0}-(\d+)'.format(label), node_lab, re.S|re.I)
                    if re_search:
                        id = int(re_search.group(1)) + 1
                
                label += '-{}'.format(id)

    return label

def transform_h2o_graph_to_python_graph(json_graph, node, graph, first_time=False):
    edges = jmespath.search("edges[?tail == `{0}`]".format(node['_gvid']), json_graph)
    edge_label_right = ''
    edge_label_left = ''

    if len(edges) > 0:
        head_left = jmespath.search("objects[?_gvid == `{0}`]".format(edges[0]['head']), json_graph)[0]
        edge_label_left = re.sub(r'\[NA\]\n', '', edges[0]['label'])
        edge_label_left = re.sub(r'\n', ',', edge_label_left)[:-1]


        head_right = jmespath.search("objects[?_gvid == `{0}`]".format(edges[1]['head']), json_graph)[0]
        edge_label_right = re.sub(r'\[NA\]\n', '', edges[1]['label'])
        edge_label_right = re.sub(r'\n', ',', edge_label_right)[:-1]
        
        keys = graph.nodes.keys()
        head_left_label = change_label_name(head_left['label'], keys)
        head_left['label'] = head_left_label

        head_right_label = change_label_name(head_right['label'], keys)
        head_right['label'] = head_right_label

        graph.add(node['label'], left_key=head_left_label, left_edge=edge_label_left, right_key=head_right_label, right_edge=edge_label_right)

        transform_h2o_graph_to_python_graph(json_graph, head_left, graph)
        transform_h2o_graph_to_python_graph(json_graph, head_right, graph)

def get_nodes_and_edges(path):
    with open(path, 'r') as a:
        json_graph = json.load(a)

    path = jmespath.search("objects[*].name", json_graph)

    occurrences = lambda s, path: (e for i, e in enumerate(path) if s in e)
    _trees = list(occurrences('cluster', path))
   
    trees = {}
    for tree in _trees:
        _nodes_ = jmespath.search("objects[?name == '{0}'].nodes".format(tree), json_graph)
        if len(_nodes_) > 0:
            if len(_nodes_[0]) > 1:
                trees[tree] = {}
                _class = re.sub(r'Tree\s*\d+,\s*Class\s*', '', jmespath.search("objects[?name == '{0}'].label".format(tree), json_graph)[0])
                trees[tree]['class'] = '' if 'Tree' in _class else _class
                trees[tree]['nodes'] = _nodes_[0]
                trees[tree]['edges'] = jmespath.search("objects[?name == '{0}'].edges".format(tree), json_graph)[0]

    return json_graph, trees

def fix_rule(rule, attributes, type_of_tree):
    rule = re.sub(r'-\d+', '', rule)

    temp_implication = rule.split(' => ')

    if type_of_tree == 'activation':
        temp_rule = temp_implication[0].split(' & ')
    else:
        temp_rule = temp_implication[1].split(' & ')

    for att in attributes:
        indexes = []
        indexes_num = []
        for bool_query in temp_rule:
            if att in bool_query:
                if '<' in bool_query or '>' in bool_query:
                    indexes_num.append(bool_query)
                else:
                    indexes.append(bool_query)
        
        if len(indexes) > 1:
            for i in indexes[:-1]:
                temp_rule.remove(i)
        
        # if len(indexes_num) > 1:
        #     rule_per_element = {'<':[], '>':[]}
        #     for element in ['<', '>']:
        #         for i, item in enumerate(indexes_num):
        #             if element in item and '{0}{1}'.format(element, '=') not in item:
        #                 rule_per_element[element].append(i)

        #     for rpe in rule_per_element.keys():
        #         if rpe == '<':
        #             if len(rule_per_element[rpe]) > 1:
        #                 min_value = float('inf')
        #                 for item in rule_per_element[rpe]:
        #                     temp_rule.remove(item)

        #                     find_float = float(re.search(r'([0-9.]+)', item, re.S|re.I).group(1))
        #                     if min_value > find_float:
        #                         min_value = find_float

        #                 temp_rule.append('{0}<={1}'.format(att, min_value))
        #         else:
        #              if len(rule_per_element[rpe]) > 1:
        #                 max_value = float('-inf')
        #                 for item in rule_per_element[rpe]:
        #                     temp_rule.remove(item)

        #                     find_float = float(re.search(r'([0-9.]+)', item, re.S|re.I).group(1))
        #                     if max_value < find_float:
        #                         max_value = find_float

        #                 temp_rule.append('{0}>={1}'.format(att, max_value))
    
    temp_rule = ' & '.join(temp_rule)


    if type_of_tree == 'activation':
        rule = temp_rule + ' => ' + temp_implication[1]
    else:
        rule = temp_implication[0] + ' => ' + temp_rule

    return rule

def fix_numerical_rules(rules, attribute, y_column_min_val, y_column_max_val, type_of_tree):
    position = -1

    if type_of_tree == 'activation':
        position = 1
    else:
        position = 0

    temp_y_rules = []
    for key in rules:
        temp_implication = rules[key].split(' => ')

        temp_y_rules.append((key, temp_implication[position]))


    for item in temp_y_rules:
        key, rule = item

        if key == len(temp_y_rules) -1:
            temp_rule = attribute + '>=' + rule.split('=')[1]
            rules[key] = rules[key].replace(rule, temp_rule)
        
        else:
            temp_rule = attribute + '<=' + rule.split('=')[1]
            rules[key] = rules[key].replace(rule, temp_rule)


    return rules

def extract_rules(performance, negative_or_positive):
    model_path = r'{0}\{1}'.format(os.path.abspath('redescription_mining/new_approach/experiment/results'), negative_or_positive)
    all_rules = []
    
    for key in performance:
        json_path = r'{0}\{1}.json'.format(model_path, key)
        json_graph, trees = get_nodes_and_edges(json_path)

        tree_graphs = []
        tree_rules = {}
        for i, tree in enumerate(trees.keys()):
            node = jmespath.search("objects[?_gvid == `{0}`]".format(trees[tree]['nodes'][0]), json_graph)[0]
            graph = BinaryTree(node['label'], type_of_tree=performance[key]['type_of_tree'], y_label=performance[key]['y_column'])

            transform_h2o_graph_to_python_graph(json_graph=json_graph, node=node, graph=graph, first_time=True)

            rules = {}
            graph.extract_rules(graph.root, rules)
            tree_graphs.append(graph)

            if trees[tree]['class'] != '':
                max_value = -1
                max_id = -1

                y_column = graph.y_label
                for rid in rules:

                    temp = re.search(r'{0}=([0-9.]+)'.format(y_column), rules[rid], re.S|re.I)
                    if temp:
                        temp = float(temp.group(1))

                        if max_value <= temp:
                            max_value = temp
                            max_id = rid

                if max_id != -1:
                    rule = rules[max_id]
                    rule = re.sub(r'{0}={1}[0-9-]*'.format(y_column, max_value), r'{0}={1}'.format(y_column, trees[tree]['class']), rule)

                    rule = fix_rule(rule, performance[key]['variable_importance'].keys(), graph.type_of_tree)

                tree_rules[i] = rule
            else:
                for rule in rules:
                    rules[rule] = fix_rule(rules[rule], performance[key]['variable_importance'].keys(), graph.type_of_tree)
                tree_rules = rules

            if performance[key]['y_column_dtype'] != 'object':
                tree_rules = fix_numerical_rules(rules=tree_rules, attribute=performance[key]['y_column'], y_column_min_val=performance[key]['y_column_min_val'], y_column_max_val=performance[key]['y_column_max_val'], type_of_tree=graph.type_of_tree)

        

        performance[key]['graph'] = tree_graphs
        performance[key]['rules'] = tree_rules
        all_rules = all_rules + list(tree_rules.values())

    _temp_r = {'rid':{}, 'query_activation': {}, 'query_target': {}}
    for i, item in enumerate(all_rules):
        rule = item.split(' => ')
        _temp_r['rid'][i] = 'r{0}'.format(i)
        _temp_r['query_activation'][i] = rule[0]
        _temp_r['query_target'][i] = rule[1]
                
    return _temp_r

def store_the_discovered_rules(redescription_data_model: RedescriptionDataModel, rules, metadata, activation_activity, target_activity):
    df = pd.DataFrame()
    rid,query_activation,query_target,acc,pval= [], [], [], [], []

    df_a, df_t = evaluate_rules_on_both_sides(redescription_data_model=redescription_data_model, rules=rules)
    
    for key in rules['rid'].keys():
        rid.append(rules['rid'][key])
        query_activation.append(rules['query_activation'][key])
        query_target.append(rules['query_target'][key])

        true_activation, true_target = support(df_a=df_a, df_t=df_t, rules=rules, position=key, metadata=metadata)

        acc.append(jaccard_index(supp_activation=true_activation, supp_target=true_target))
        pval.append(p_value(supp_activation=true_activation, supp_target=true_target, E=df_a.shape[0]))
            
    df['rid'] = rid
    df['query_activation'] = query_activation
    df['query_target'] = query_target
    df['acc'] = acc
    df['pval'] = pval
    df['card_Exo'] = None
    df['card_Eox'] = None
    df['card_Exx'] = None
    df['card_Eoo'] = None
    df['activation_vars'] = ', '.join(redescription_data_model.activation_attributes)
    df['activation_activity'] = activation_activity
    df['target_vars'] = ', '.join(redescription_data_model.target_attributes)
    df['target_activity'] = target_activity

    return df
