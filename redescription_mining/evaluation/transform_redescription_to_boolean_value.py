import re
import pandas as pd

def solve_string_expression(s):
        stack = []
        op = {
            "or": lambda x, y: x or y,
            "and": lambda x, y: x and y,
            "|": lambda x, y: x or y,
            "&": lambda x, y: x and y
        }
        s = re.sub(r'\(\s+', '(', s)
        s = re.sub(r'\s+\)', ')', s)
        
        elements = s.split()
        parentheses_not = False
        parentheses = False
        pass_i = -1
        for i, v in enumerate(elements):
            if i == len(elements) - 1 and parentheses:
                if v.count(")") > 0:
                    if type(stack[index][len(stack[index])-1]) != bool:
                        stack[index].append(v[0] == "T")
                    right = stack[index].pop()
                    for _ in range(int(len(stack[index])/2)):
                        o = stack[index].pop()
                        left = stack[index].pop()
                        right = o(left, right)
                    
                    if parentheses_not:
                        stack.append(not right)
                    else:
                        stack.append(right)

                    stack.remove([])
                    parentheses_not = False
                    parentheses = False
            elif i <= pass_i:
                pass
            else:
                if v[0] == "(":
                    stack.append([])
                    index = len(stack) - 1
                    if v[v.count("("):] == 'not' or v[v.count("("):] == '!':
                        stack[index].append(not (elements[i+1] == "T"))
                            
                        if ')' not in elements[i+1]:
                            pass_i = i + 1
                    else:
                        stack[index].append(v[v.count("("):] == "T")

                    parentheses = True

                elif v.count(")") > 0:
                    if type(stack[index][len(stack[index])-1]) != bool:
                        stack[index].append(v[0] == "T")
                    right = stack[index].pop()
                    for _ in range(int(len(stack[index])/2)):
                        o = stack[index].pop()
                        left = stack[index].pop()
                        right = o(left, right)
                    
                    if parentheses_not:
                        stack.append(not right)
                    else:
                        stack.append(right)

                    stack.remove([])
                    parentheses_not = False
                    parentheses = False

                elif v in ["T", "F"]:
                    if parentheses:
                        stack[index].append(v == "T")
                    else:
                        stack.append(v == "T")
                else:
                    if v == 'not' or v == '!':
                        if "(" in elements[i+1]:                            
                            parentheses_not = True
                        else:
                            if parentheses:
                                stack[index].append(not (elements[i+1] == "T"))
                            else:
                                stack.append(not (elements[i+1] == "T"))
                            
                            if ')' not in elements[i+1]:
                                pass_i = i + 1
                        
                    else:
                        if parentheses:
                            stack[index].append(op[v])
                        else:
                            stack.append(op[v])

        if len(stack) > 1:
            for i in range(0, len(stack) - 1, 2):
                stack[i + 2] = stack[i + 1](stack[i], stack[i + 2])
            return stack[-1]

        return stack[0]

def evaluate_subrules(rule):
    elements = rule.strip().split()

    for index, item in enumerate(elements):
        if '<=' in item:   
            temp = item.split('<=')
            elements[index] = "T" if float(temp[0].strip()) <= float(temp[1].strip()) else "F" 
        elif '<' in item:   
            temp = item.split('<')
            elements[index] = "T" if float(temp[0].strip()) < float(temp[1].strip()) else "F" 

        elif '>=' in item:   
            temp = item.split('>=')
            elements[index] = "T" if float(temp[0].strip()) >= float(temp[1].strip()) else "F" 
        elif '>' in item:   
            temp = item.split('>')
            elements[index] = "T" if float(temp[0].strip()) > float(temp[1].strip()) else "F" 
        elif '=' in item:
            temp = item.split('=')
            elements[index] = "T" if temp[0].strip() == temp[1].strip() else "F"

    return solve_string_expression((' '.join(elements)).strip())

def apply_implication(activation_rule, target_rule, type='implication'):
    if type == 'similarity':
        return activation_rule == target_rule
    else:
        if activation_rule and not target_rule:
            return False
        else:
            return True

def rules_to_boolean_value(df_a, df_t, rules, for_deviant_traces=False):
    df = pd.DataFrame()
    replace_variables = lambda x, y: x.replace(col, 'T' if str(y).lower() == 'true' else 'F' if str(y).lower() == 'false' else str(y))
    _evaluate_subrules = lambda x: evaluate_subrules(x)
    _apply_implication = lambda x, y: apply_implication(x, y)

    for key in rules['rid'].keys():
        rid = rules['rid'][key]
        rule_left = rules['query_activation'][key]
        rule_right = rules['query_target'][key]

        df_a[rid] = rule_left
        for col in df_a.columns:
            if col in rule_left:
                df_a[rid] = df_a[rid].combine(df_a[col], func=replace_variables)


        df_t[rid] = rule_right
        for col in df_t.columns:
            if col in rule_right:
                df_t[rid] = df_t[rid].combine(df_t[col], func=replace_variables)
        
        df_a[rid] = df_a[rid].apply(func=_evaluate_subrules)
        df_t[rid] = df_t[rid].apply(func=_evaluate_subrules)

        df[rid] = df_a[rid].combine(df_t[rid], func=_apply_implication)

    if for_deviant_traces:
        return df
    else:
        return df_a, df_t

