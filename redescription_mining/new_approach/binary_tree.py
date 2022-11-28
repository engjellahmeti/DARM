"""
    @Author: Engjëll Ahmeti
    @Date: 11.11.2022
    @LastUpdate: 21.11.2022
"""
import re

class BinaryTreeNode:
    def __init__(self, key, left=None, left_edge=None, right=None, right_edge=None, type_of_tree=None, y_label=None):
        self.key = key
        self.left_node = left
        self.left_edge = left_edge
        self.right_node = right
        self.right_edge = right_edge
        self.type_of_tree = type_of_tree
        self.y_label = y_label

class BinaryTree:
    def __init__(self, root_key=None, type_of_tree=None, y_label=None):
        self.nodes = {}
        self.type_of_tree = type_of_tree
        self.y_label = y_label
        if root_key is not None:
            self.root = BinaryTreeNode(root_key, type_of_tree=type_of_tree, y_label=y_label)
            self.nodes[root_key] = self.root

    def add(self, key, left_key=None, left_edge=None, right_key=None, right_edge=None):
        if key not in self.nodes:
            self.nodes[key] = BinaryTreeNode(key)

        if left_key is None:
            self.nodes[key].left_node = None
        else:
            if left_key not in self.nodes:
                self.nodes[left_key] = BinaryTreeNode(left_key, type_of_tree=self.type_of_tree, y_label=self.y_label)
            self.nodes[key].left_edge = left_edge
            self.nodes[key].left_node = self.nodes[left_key]

        if right_key == None:
            self.nodes[key].right_node = None
        else:
            if right_key not in self.nodes:
                self.nodes[right_key] = BinaryTreeNode(right_key, type_of_tree=self.type_of_tree, y_label=self.y_label)
            self.nodes[key].right_edge = right_edge
            self.nodes[key].right_node = self.nodes[right_key]
    
    def extract_rules(self, node, rules, id=0):
        left_id = id
        right_id = -1
        if node is not None:
            left_node = node.left_node
            right_node = node.right_node

            node_key = node.key
            if '<' in node_key or '>' in node_key:
                if '>' in node_key:
                    print('method extract_rules in the binary tree just met > symbol')

                node_key = re.sub(r'\s+', '', node_key)
                temp_num = ''
                if id in rules.keys():
                    temp_num = rules[id]
                    rules[id] += ' & ' + node_key
                else:
                    rules[id] = node_key

                split_node_key = node_key.split('<')
                last_num = int(split_node_key[1][-1]) - 1
                if last_num == -1:
                    if split_node_key[1][-2] == '.':
                        temp_split_node_key = int(split_node_key[1][0:-2]) - 1
                        temp_key = str(temp_split_node_key) + '.9'
                else:
                    temp_key = split_node_key[1][0:-1] + str(last_num)
                    
                text = temp_key + '<' + split_node_key[0]

                right_id = len(rules.keys())
                if right_id in rules.keys():
                    rules[right_id] += ' & ' + text
                else:
                    if temp_num != '':
                        rules[right_id] = temp_num + ' & ' + text
                    else:
                        rules[right_id] = text

            else:
                text = ''
                temp = ''
                if id in rules.keys():
                    temp = rules[id]
                if left_node:
                    if ',' not in node.left_edge:
                        if node.left_edge.lower() == 'true':
                            text = node.key
                        elif node.left_edge.lower() == 'false':
                            text = '! ' + node.key
                        else:
                            text = node.key + '=' + node.left_edge

                    else:
                        text = '( '
                        for item in node.left_edge.split(','):
                            text += node.key + '=' + item + ' | '
                        text = text[:-3] + ' )'

                    if id in rules.keys():
                        if text != '':
                            rules[id] = rules[id] + ' & ' + text
                    else:
                        rules[id] = text
            
                text = ''
                if node.right_edge:
                    if ',' not in node.right_edge:
                        if node.right_edge.lower() == 'true':
                            text = node.key
                        elif node.right_edge.lower() == 'false':
                            text = '! ' + node.key
                        else:
                            text = node.key + '=' + node.right_edge

                    else:
                        text = '( '
                        for item in node.right_edge.split(','):
                            text += node.key + '=' + item + ' | '
                        text = text[:-3] + ' )'

                    right_id = len(rules.keys())
                    if text != '':
                        rules[right_id] = temp + ' & ' + text


            if left_node is None and right_node is None:
                if id in rules.keys():
                    if rules[id][1] == '&':
                        rules[id] = rules[id][3:]
                        
                    if self.type_of_tree == 'activation':
                        rules[id] += ' => ' + node.y_label + '=' + node.key
                    else:
                        rules[id] = node.y_label + '=' + node.key  + ' => ' + rules[id]
            else:
                self.extract_rules(left_node, rules, left_id)
                self.extract_rules(right_node, rules, right_id)
