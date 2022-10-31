# @project Deviance Analysis by Means of Redescription Mining - Master Thesis 
# @author Engjëll Ahmeti
# @date 1/23/2021
import pandas as pd
import re
from nlg.constraint import Constraint
from nlg.conjunction import Conjunction
from nlg.disjunction import Disjunction
from nlg.parentheses import Parentheses
from nlg.negation import Negation
from nlg.implication import Implication
from nlg.extraction import ExtractionConstraint
from nlg.comparison import Comparison
import stanza

class NLG:
    def __init__(self, extract_dsynts_on_leafs):
        self.operators = {'|': 'or', '&': 'and'}
        self.nlp = stanza.Pipeline('en')
        self.extract_dsynts_on_leafs = extract_dsynts_on_leafs
        self.activationOrTarget = ''

    def findRulesThatCorrespond(self, negativePath, positivePath):
        negative = pd.read_csv(negativePath)
        positive = pd.read_csv(positivePath)
        negative.drop(['acc', 'pval', 'card_Exo', 'card_Eox', 'card_Exx', 'card_Eoo'], axis=1, inplace=True)
        positive.drop(['acc', 'pval', 'card_Exo', 'card_Eox', 'card_Exx', 'card_Eoo'], axis=1, inplace=True)

        n = []
        p = []
        for index, item in enumerate(negative.columns):
            n.append(item + '-negative')
            p.append(positive.columns[index] + '-positive')

        listOfRules = n + p

        final = pd.DataFrame(columns=listOfRules)

        index = 0
        for row in positive.iterrows():
            row = row[1]
            for row_neg in negative.iterrows():
                row_neg = row_neg[1]

                if set(sorted(row_neg['activation_vars'])).issubset(sorted(row['activation_vars'])) and set(
                        sorted(row_neg['target_vars'])).issubset(sorted(row['target_vars'])):
                    listOfRules = list(row_neg[negative.columns])
                    listOfRules1 = list(row[positive.columns])
                    listOfRules2 = listOfRules + listOfRules1
                    final.loc[index] = listOfRules2
                    index += 1

        return final

    def findParentheses(self, listOfRules):
        parentheses = []
        index1 = -1
        in_ = False
        parentheses_closed = False
        for index, rule in enumerate(listOfRules):
            if index > index1:
                if '(' == rule:
                    parentheses_closed = False
                    in_ = True
                    if index > 0 and len(parentheses) == 0:
                        parentheses = parentheses + listOfRules[0:index]
                    index1 = 0
                    for index1, rule1 in enumerate(listOfRules[index:len(listOfRules)]):
                        index1 = index1 + index
                        if index1 > index and ')' == rule1 and not parentheses_closed:
                            parentheses.append(listOfRules[index:index1 + 1])
                            parentheses_closed = True
                            break

                elif rule in self.operators.keys() and in_:
                    parentheses.append(rule)

                elif parentheses_closed:
                    for item in ['=', '<', '>']:
                        if item in rule:
                            parentheses.append(rule)

        return parentheses

    def extractParentheses(self, attributes, listOfRules, activity):
        ruleGenerated = None

        for index, rule in enumerate(listOfRules):
            if len(listOfRules) == 1:
                if '(' not in listOfRules[0]:
                    constraint_ = self.extractConstraint(attributes, rule, activity)
                else:
                    constraint_ = self.extractParentheses(attributes, rule, activity)

            if rule in self.operators.keys():
                if not (isinstance(listOfRules[index - 1], Conjunction) or isinstance(listOfRules[index - 1],
                                                                                      Disjunction)):
                    if '(' not in listOfRules[index - 1]:
                        constraint1 = self.extractConstraint(attributes, listOfRules[index - 1], activity)
                    else:
                        constraint1 = self.extractParentheses(attributes, listOfRules[index - 1], activity)

                else:
                    constraint1 = listOfRules[index - 1]

                if '(' not in listOfRules[index + 1]:
                    constraint2 = self.extractConstraint(attributes, listOfRules[index + 1], activity)
                else:
                    constraint2 = self.extractParentheses(attributes, listOfRules[index + 1], activity)

                if rule == '|':
                    disjunction_ = Disjunction(constraint1, constraint2, self.extract_dsynts_on_leafs)
                    listOfRules[index + 1] = disjunction_
                    ruleGenerated = disjunction_
                elif rule == '&':
                    conjunction_ = Conjunction(constraint1, constraint2, self.extract_dsynts_on_leafs)
                    listOfRules[index + 1] = conjunction_
                    ruleGenerated = conjunction_

        return Parentheses(ruleGenerated, self.extract_dsynts_on_leafs)

    def extractConstraint(self, attributes, rule, activity):
        attribute = None
        for att in attributes:
            if att in rule:
                attribute = att

        first = None
        second = None
        constraint_ = None

        if attribute is None:
            print()

        negation = False
        if '!' in rule:
            negation = True

        if re.search(r'([0-9.]+)[<>]+' + attribute + '[<>]+([0-9.]+)', rule, re.S | re.I):
            first = re.search(r'([0-9.]+)[<>]+' + attribute + '[<>]+([0-9.]+)', rule, re.S | re.I).group(1)
            second = re.search(r'([0-9.]+)[<>]+' + attribute + '[<>]+([0-9.]+)', rule, re.S | re.I).group(2)

            constraint_ = Constraint(attribute, first, second, '<', activity, self.nlp, negation, self.extract_dsynts_on_leafs, self.activationOrTarget)

        elif re.search(r'([0-9.]+)[<>]+' + attribute, rule, re.S | re.I):
            first = re.search(r'([0-9.]+)[<>]+' + attribute, rule, re.S | re.I).group(1)
            constraint_ = Constraint(attribute, first, None, '<', activity, self.nlp, negation, self.extract_dsynts_on_leafs, self.activationOrTarget)


        elif re.search(attribute + '[<>]+([0-9.]+)', rule, re.S | re.I):
            second = re.search(attribute + '[<>]+([0-9.]+)', rule, re.S | re.I).group(1)
            constraint_ = Constraint(attribute, None, second, '<', activity, self.nlp, negation, self.extract_dsynts_on_leafs, self.activationOrTarget)

        elif re.search(attribute + '=([A-Za-z0-9.]*)', rule, re.S | re.I):
            first = re.search(attribute + '=([A-Za-z0-9.]*)', rule, re.S | re.I).group(1)
            constraint_ = Constraint(attribute, first, None, '=', activity, self.nlp, negation, self.extract_dsynts_on_leafs, self.activationOrTarget)

        elif re.search(attribute, rule, re.S | re.I):
            constraint_ = Constraint(attribute, None, None, 'boolean', activity, self.nlp, negation, self.extract_dsynts_on_leafs, self.activationOrTarget)

        if negation:
            constraint_ = Negation(constraint_, self.extract_dsynts_on_leafs)

        return constraint_

    def setupObjects(self, ruleInString, attributes, activity, activationOrTarget):
        listOfRules = re.split('([\|\&\(\)])', ruleInString)
        listOfRules = list(filter(lambda a: a != '', listOfRules))
        listOfRules = list(filter(lambda a: a != ' ', listOfRules))
        listOfRules = [str(x).strip() for x in listOfRules]
        self.activationOrTarget = activationOrTarget
        if '(' in listOfRules:
            listOfRules = self.findParentheses(listOfRules)

        ruleGenerated = None

        for index, rule in enumerate(listOfRules):
            if len(listOfRules) == 1:
                if '(' not in listOfRules[0]:
                    constraint_ = self.extractConstraint(attributes, rule, activity)
                else:
                    constraint_ = self.extractParentheses(attributes, rule, activity)
                ruleGenerated = constraint_

            if not isinstance(rule, list) and rule in self.operators.keys():
                if not (isinstance(listOfRules[index - 1], Conjunction) or isinstance(listOfRules[index - 1],
                                                                                      Disjunction) or isinstance(
                        listOfRules[index - 1], Parentheses) or isinstance(listOfRules[index - 1], Negation)):
                    if '(' not in listOfRules[index - 1]:
                        constraint1 = self.extractConstraint(attributes, listOfRules[index - 1], activity)
                    else:
                        constraint1 = self.extractParentheses(attributes, listOfRules[index - 1], activity)

                else:
                    constraint1 = listOfRules[index - 1]

                if '(' not in listOfRules[index + 1]:
                    constraint2 = self.extractConstraint(attributes, listOfRules[index + 1], activity)
                else:
                    constraint2 = self.extractParentheses(attributes, listOfRules[index + 1], activity)

                if rule == '|':
                    disjunction_ = Disjunction(constraint1, constraint2, self.extract_dsynts_on_leafs)
                    listOfRules[index + 1] = disjunction_
                    ruleGenerated = disjunction_
                elif rule == '&':
                    conjunction_ = Conjunction(constraint1, constraint2, self.extract_dsynts_on_leafs)
                    listOfRules[index + 1] = conjunction_
                    ruleGenerated = conjunction_

        return ruleGenerated

    def transform_conll_to_dsynts(self, setOfRules):
        listOfDSyntS = []
        for item in setOfRules:
            extract = ExtractionConstraint(item[0].__str__(), self.nlp, 'imply', True)

            listPos = []
            for itemP in item[1]:
                extract = ExtractionConstraint(itemP.__str__(), self.nlp, 'imply', True)
                listPos.append(extract.DSyntS)

            listOfDSyntS.append((extract.DSyntS, listPos))

        return listOfDSyntS

    def apply_comparisons(self, set_of_rules, filename=None):
        comparison = Comparison(set_of_rules, filename)
        return comparison.compare()

    def find_deviant_traces(self, filename):
        comparison = Comparison(None)
        if 'splittrees' in filename:
            return comparison.deviantTracesSplitTrees(filename, self.nlp)
        else:
            return comparison.deviantTraces_v2(filename, self.nlp)

    def nlg(self, negativePath, positivePath):

        final = self.findRulesThatCorrespond(negativePath, positivePath)
        groups = final.groupby(
            ['query_activation-negative', 'activation_vars-negative', 'activation_activity-negative', 'query_target-negative',
             'target_vars-negative', 'target_activity-negative','constraint-negative'])

        setOfRules = []
        redescriptions = []
        for name, group in groups:
            groupOfRules = pd.DataFrame(group, columns=group.columns)
            negativeNLGLeft = self.setupObjects(name[0], name[1].split(','), name[2], 'activation')
            negativeNLGRight = self.setupObjects(name[3], name[4].split(','), name[5], 'target')

            negativeRule = Implication(negativeNLGLeft, negativeNLGRight, self.extract_dsynts_on_leafs, name[6], name[0], name[3], groupOfRules.iloc[0]['rid-negative'])

            positiveRules = []
            redescriptionsPos = []
            for row in groupOfRules.iterrows():
                row = row[1]

                positiveNLGLeft = self.setupObjects(row['query_activation'
                                                        '-positive'], row['activation_vars-positive'].split(','),
                                                    row['activation_activity-negative'], 'activation')

                positiveNLGRight = self.setupObjects(row['query_target-positive'],
                                                     row['target_vars-positive'].split(','), row['target_activity-negative'], 'target')

                positiveRule = Implication(positiveNLGLeft, positiveNLGRight, self.extract_dsynts_on_leafs, row['constraint-negative'], row['query_activation-positive'], row['query_target-positive'], row['rid-positive'])

                positiveRules.append(positiveRule)
                redescriptionsPos.append(row['query_activation-positive'] + ' => ' + row['query_target-positive'])

            setOfRules.append((negativeRule, positiveRules))
            redescriptions.append((name[0] + ' => ' + name[3], redescriptionsPos))



        return (setOfRules, redescriptions)

    def nlgSplit(self, negativePath, positivePath):
        negRules = []
        posRules = []

        negative = pd.read_csv(negativePath)
        positive = pd.read_csv(positivePath)

        for row in negative.iterrows():
            row = row[1]
            left = self.setupObjects(row['query_activation'], row['activation_vars'].split(','),
                                     row['activation_activity'], 'activation')

            right = self.setupObjects(row['query_target'],
                                      row['target_vars'].split(','), row['target_activity'],
                                      'target')

            rule = Implication(left, right, self.extract_dsynts_on_leafs,
                               row['constraint'], row['query_activation'], row['query_target'])

            negRules.append((rule, row['rid'], row['query_activation'] + ' => ' + row['query_target']))

        for row in positive.iterrows():
            row = row[1]
            left = self.setupObjects(row['query_activation'], row['activation_vars'].split(','),
                                     row['activation_activity'], 'activation')

            right = self.setupObjects(row['query_target'],
                                      row['target_vars'].split(','), row['target_activity'],
                                      'target')

            rule = Implication(left, right, self.extract_dsynts_on_leafs,
                               row['constraint'], row['query_activation'], row['query_target'])

            posRules.append((rule, row['rid'], row['query_activation'] + ' => ' + row['query_target']))

        return (negRules, posRules)
