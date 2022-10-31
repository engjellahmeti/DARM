# @project Deviance Analysis by Means of Redescription Mining - Master Thesis 
# @author Engjëll Ahmeti
# @date 2/2/2021
from log_print import Print
import re
import pandas as pd
import stanza
import random as rd
import os
import math

class Comparison:
    def __init__(self, setOfRules, filename=None):
        self.setOfRules = setOfRules
        self.listt = []
        self.dictRule = {}
        self.filename = filename
        self.ruleIDExtract = None

    def compare(self):
        negListt = []
        posListt = []
        days = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eigth',
                9: 'ninth', 10: 'tenth', 11: 'eleventh'}
        Print.GREEN.print(
            'Comparisons between the set of negative and positive redescription rules that relate on the attribute names, but have different nature: ')
        index = 1
        output = 'Comparisons between the set of negative and positive redescription rules that relate on the attribute names, but have different nature:\n'


        for row in self.setOfRules:
            ruleId = row[0].ruleID
            if ruleId is not None:
                Print.END.print('The {1} mined negative rule  is \'{0} ({2})\' and its subrules comparisons to the positive subrules are below:'.format(Print.RED.__call__(row[0].__str__()), days[index], ruleId))
                output +=  'The {1} mined negative rule  is \'{0} ({2})\' and its subrules comparisons to the positive subrules are below:\n'.format(row[0].__str__(), days[index], ruleId)
            else:
                Print.END.print('The {1} mined negative rule is \'{0}\' and its subrules comparisons to the positive subrules are below:'.format(Print.RED.__call__(row[0].__str__()), days[index]))
                output += 'The {1} mined negative rule is \'{0}\' and its subrules comparisons to the positive subrules are below:\n'.format(row[0].__str__(), days[index])

            self.extractSubrules(row[0].tree, row[0].__str__())
            negListt = self.listt
            self.listt = []

            index += 1

            temp = ''
            dictPositive = {}
            for rowP in row[1]:
                temp = '\n           - {0}'.format(rowP.__str__())
                self.ruleIDExtract = rowP.ruleID
                self.extractSubrules(rowP.tree, row[0].__str__())
                posListt = posListt + self.listt
                dictPositive[self.ruleIDExtract] = posListt
                self.listt = []
                self.ruleIDExtract = None


            listOfDifferences = []
            for rule in dictPositive.keys():
                for itemPos in dictPositive[rule]:
                    for item in negListt:
                        negKey = list(item.keys())[0]
                        posKey = list(itemPos.keys())[0]

                        if negKey == posKey and item[negKey]['activity'] == itemPos[posKey]['activity'] and itemPos[posKey]['activationOrTarget'] == itemPos[negKey]['activationOrTarget']:
                            activity = re.sub(r"(\w)([A-Z])", r"\1 \2", item[negKey]['activity'])
                            attribute = re.sub(r"(\w)([A-Z])", r"\1 \2", negKey)

                            listOfDifferences = self.extractAttributesWithDifferentValues(listOfDifferences, activity, attribute, item, itemPos, negKey, posKey)        

            posListt = []
            # Print.END.print('The output')
            for item in listOfDifferences:
                negativeContent = re.search(r'{0}\s*(.*$)'.format(item[1].lower()), item[2], re.S | re.I)
                if negativeContent:
                    negativeContent = negativeContent.group(1)
                elif 'executed' in item[2]:
                    negativeContent = re.sub('the process ', '', item[2])

                positiveContent = re.search(r'{0}\s*(.*$)'.format(item[1].lower()), item[3], re.S | re.I)
                if positiveContent:
                    positiveContent = positiveContent.group(1)
                elif 'executed' in item[3]:
                    positiveContent = re.sub('the process ', '', item[3])

                Print.END.print('          - The {0} for the event \'{1}\' {2}in the negative rule, while in the positive rule {5}, it {3}.'.format(item[1].lower(), Print.BLUE.__call__(item[0]), negativeContent, (positiveContent).strip(), Print.RED.__call__(item[5]), item[6]) )
                output += '          - The {0} for the event \'{1}\' {2}in the negative rule, while in the positive rule {5}, it {3}.\n'.format(item[1].lower(), item[0], negativeContent, (positiveContent).strip(), item[5], item[6])
            print()
            output += '\n'
        
        return output

    def extractAttributesWithDifferentValues(self, listOfDifferences, activity, attribute, item, itemPos, negKey, posKey):
        if '(' in item[negKey]['interval'] and '(' in itemPos[negKey]['interval']:
            negInterval = re.sub('\s|\)|\(', '', item[negKey]['interval']).split(',')
            posInterval = re.sub('\s|\)|\(', '', itemPos[negKey]['interval']).split(',')

            start_neg = int(negInterval[0]) if '-∞' != negInterval[0] else -math.inf
            end_eng =  int(negInterval[1]) if '+∞' != negInterval[1] else math.inf

            start_pos = int(posInterval[0]) if '-∞' != posInterval[0] else -math.inf
            end_pos =  int(posInterval[1]) if '+∞' != posInterval[1] else math.inf


            if not (start_pos <= start_neg and end_eng <= end_pos):
                if (activity, attribute, item[negKey]['output'], itemPos[posKey]['output'], itemPos[negKey]['activationOrTarget'], item[negKey]['fullOutput'], itemPos[negKey]['ruleID']) not in listOfDifferences:
                    listOfDifferences.append((activity, attribute, item[negKey]['output'], itemPos[posKey]['output'], itemPos[negKey]['activationOrTarget'], item[negKey]['fullOutput'], itemPos[negKey]['ruleID']))
            

        elif item[negKey]['interval'] != itemPos[negKey]['interval']:
            activity = re.sub(r"(\w)([A-Z])", r"\1 \2", item[negKey]['activity'])
            attribute = re.sub(r"(\w)([A-Z])", r"\1 \2", negKey)

            if (activity, attribute, item[negKey]['output'], itemPos[posKey]['output'], itemPos[negKey]['activationOrTarget'], item[negKey]['fullOutput'], itemPos[negKey]['ruleID']) not in listOfDifferences:
                listOfDifferences.append((activity, attribute, item[negKey]['output'], itemPos[posKey]['output'], itemPos[negKey]['activationOrTarget'], item[negKey]['fullOutput'], itemPos[negKey]['ruleID']))

        return listOfDifferences                    

    def extractSubrules(self, tree, fullOutput):
        if type(tree) is list:
            for row in tree:
                for key in row.keys():
                    if key not in ['imply', 'or', 'and', 'not', 'parentheses']:
                        temp = row.copy()
                        temp[key]['fullOutput'] = fullOutput
                        temp[key]['ruleID'] = self.ruleIDExtract
                        self.listt.append(temp)

                    else:
                        self.extractSubrules(row[key], fullOutput)

        else:
            for key in tree.keys():
                if key not in ['imply', 'or', 'and', 'not', 'parentheses']:
                    temp = tree.copy()
                    temp[key]['fullOutput'] = fullOutput
                    temp[key]['ruleID'] = self.ruleIDExtract
                    self.listt.append(temp)
                else:
                    self.extractSubrules(tree[key], fullOutput)

    def deviantTraces(self, positiveRules, nlp):
        merged = pd.read_csv('../feature_vectors/csv_feature_vectors/negative/traces.csv', index_col=0)

        positiveRules = pd.read_csv(r'../redescription_mining/results/final-reremi-positive.queries')

        full = []
        dict = {}
        for rule in positiveRules.iterrows():
            rule = rule[1]
            rid = rule['rid']



            attributesInRule = self.extract(rule['query_activation'], rule['activation_vars'].split(',')) + self.extract(rule['query_target'], rule['target_vars'].split(','))
            # frameReturned = self.solveRuleForFrame(rule, merged, attributesInRule)

            for row in merged.iterrows():
                row = row[1]
                frameReturned = self.solveRuleForRow(rule, row, attributesInRule)

                for item in attributesInRule:
                        if len(item[1]) == 1:
                            if item[1][0][1] == False:
                                if row[item[0]] != item[1][0][0]:
                                    full.append([rid, row['Trace']])

                                    attribute = item[0]
                                    activity = self.findActivity(rule, attribute)
                                    if row['Trace'] not in dict.keys():
                                        dict[row['Trace']] = [self.outputNLG(rid, nlp, activity, attribute, item[1][0][2], item[1][0][0], None)]
                                    else:
                                        dict[row['Trace']].append(self.outputNLG(rid, nlp, activity, attribute, item[1][0][2], item[1][0][0], None))
                            else:
                                if row[item[0]] == item[1][0][0]:
                                    full.append([rid, row['Trace']])
                                    attribute = item[0]
                                    activity = self.findActivity(rule, attribute)

                                    if row['Trace'] not in dict.keys():
                                        dict[row['Trace']] = [self.outputNLG(rid, nlp, activity, attribute, item[1][0][2], item[1][0][0], None)]
                                    else:
                                        dict[row['Trace']].append(self.outputNLG(rid, nlp, activity, attribute, item[1][0][2], item[1][0][0], None))

                        else:
                            if None in item[1]:
                                if item[1][0] is None:
                                    if row[item[0]] >= int(item[1][1]):
                                        full.append([rid, row['Trace']])

                                        attribute = item[0]
                                        activity = self.findActivity(rule, attribute)

                                        if row['Trace'] not in dict.keys():
                                            dict[row['Trace']] = [self.outputNLG(rid, nlp, activity, attribute, item[1][2], None, item[1][1])]
                                        else:
                                            dict[row['Trace']].append(self.outputNLG(rid, nlp, activity, attribute, item[1][2], None, item[1][1]))

                                else:
                                    if row[item[0]] <= int(item[1][0]):
                                        full.append([rid, row['Trace']])

                                        attribute = item[0]
                                        activity = self.findActivity(rule, attribute)

                                        if row['Trace'] not in dict.keys():
                                            dict[row['Trace']] = [
                                                self.outputNLG(rid, nlp, activity, attribute, item[1][2], item[1][0], None)]
                                        else:
                                            dict[row['Trace']].append(
                                                self.outputNLG(rid, nlp, activity, attribute, item[1][2], item[1][0], None))


                            else:
                                if row[item[0]] <= int(item[1][0]) or row[item[0]] >= int(item[1][1]):
                                    full.append([rid, row['Trace']])

                                    attribute = item[0]
                                    activity = self.findActivity(rule, attribute)

                                    if row['Trace'] not in dict.keys():
                                        dict[row['Trace']] = [
                                            self.outputNLG(rid, nlp, activity, attribute, item[1][2], item[1][0], item[1][1])]
                                    else:
                                        dict[row['Trace']].append(
                                            self.outputNLG(rid, nlp, activity, attribute, item[1][2], item[1][0], item[1][1]))

            print()

    def deviantTraces_v2(self, positiveRules, nlp):
        merged_ = pd.read_csv(os.path.abspath('feature_vectors/csv_feature_vectors/negative/traces.csv'), index_col=0)
        positiveRules = pd.read_csv(os.path.abspath('redescription_mining/results/'+positiveRules+'-positive.queries'))

        merged = merged_.loc[rd.choices(list(merged_.index.values), k=7)]
        mergedColumns = ','.join(list(merged.columns))
        full = []
        dict = {}
        for rule in positiveRules.iterrows():
            rule = rule[1]
            rid = rule['rid']
            leftRule_ = re.sub('\|', 'or', rule['query_activation'])
            leftRule_ = re.sub('\&', 'and', leftRule_)

            rightRule_ = re.sub('\|', 'or', rule['query_target'])
            rightRule_ = re.sub('\&', 'and', rightRule_)

            if '_x' in mergedColumns and '_y' in mergedColumns:
                vars = re.findall(r'([A-Za-z]+)_', mergedColumns, re.S|re.I)
                if vars:
                    vars = set(vars)
                    leftQuery = rule['query_activation']
                    leftVars = rule['activation_vars']

                    rightQuery = rule['query_target']
                    rightVars = rule['target_vars']
                    for var in vars:
                        leftQuery = re.sub(var, var + '_x', leftQuery)
                        leftVars = re.sub(var, var + '_x', leftVars)

                        rightQuery = re.sub(var, var + '_y', rightQuery)
                        rightVars = re.sub(var, var + '_y', rightVars)

                    leftRule_ =  re.sub('\|', 'or', leftQuery)
                    leftRule_ =  re.sub('\&', 'and', leftRule_)

                    rightRule_ = re.sub('\|', 'or',rightQuery)
                    rightRule_ = re.sub('\&', 'and', rightRule_)

                    attributesInRule = self.extract(leftQuery, leftVars.split(',')) + self.extract(rightQuery, rightVars.split(','))

            else:
                attributesInRule = self.extract(rule['query_activation'], rule['activation_vars'].split(',')) + self.extract(rule['query_target'], rule['target_vars'].split(','))


            for row in merged.iterrows():
                row = row[1]
                # leftRule = re.sub('\|', 'or', rule['query_activation'])
                # leftRule = re.sub('\&', 'and', leftRule)
                #
                # rightRule = re.sub('\|', 'or', rule['query_target'])
                # rightRule = re.sub('\&', 'and', rightRule)
                leftRule = leftRule_
                rightRule = rightRule_

                breakItemNotInFrame = True
                for item in attributesInRule:
                    if item[0] in mergedColumns:
                        var = item[0]
                        if len(item[1]) == 1:
                            if item[1][0][1] == False:
                                if row[item[0]] != item[1][0][0]:
                                    full.append([rid, row['Trace']])

                                    attribute = item[0]
                                    activity = self.findActivity(rule, attribute)

                                    dict = self.outputNLG(rid, nlp, activity, attribute, item[1][0][2], item[1][0][0], None, dict, row['Trace'])

                                    # if var in leftRule:
                                    #     leftRule = re.sub(var + '=' + str(item[1][0][0]), 'F', leftRule)
                                    # else:
                                    #     rightRule = re.sub(var + '=' + str(item[1][0][0]), 'F', rightRule)

                                    (leftRule, rightRule) = self.dealWithEqualnessAndBoolean(item[1][0][1], True, var, leftRule, rightRule, item[1][0][0])
                                else:
                                    # if var in leftRule:
                                    #     leftRule = re.sub(var + '=' + str(item[1][0][0]), 'T', leftRule)
                                    # else:
                                    #     rightRule = re.sub(var + '=' + str(item[1][0][0]), 'T', rightRule)
                                    (leftRule, rightRule) = self.dealWithEqualnessAndBoolean(item[1][0][1], False, var, leftRule, rightRule, item[1][0][0])

                            else:
                                if row[item[0]] == item[1][0][0]:

                                    full.append([rid, row['Trace']])
                                    attribute = item[0]
                                    activity = self.findActivity(rule, attribute)

                                    dict = self.outputNLG(rid, nlp, activity, attribute, item[1][0][2], item[1][0][0], None, dict, row['Trace'])

                                    # if var in leftRule:
                                    #     leftRule = re.sub('!\s*' + var + '=' + str(item[1][0][0]), 'F', leftRule)
                                    # else:
                                    #     rightRule = re.sub('!\s*' + var + '=' + str(item[1][0][0]), 'F', rightRule)

                                    (leftRule, rightRule) = self.dealWithEqualnessAndBoolean(item[1][0][1], False, var, leftRule, rightRule, item[1][0][0])

                                else:
                                    # if var in leftRule:
                                    #     leftRule = re.sub('!\s*' + var + '=' + str(item[1][0][0]), 'T', leftRule)
                                    # else:
                                    #     rightRule = re.sub('!\s*' + var + '=' + str(item[1][0][0]), 'T', rightRule)
                                    (leftRule, rightRule) = self.dealWithEqualnessAndBoolean(item[1][0][1], True, var,
                                                                                                 leftRule, rightRule,
                                                                                                 item[1][0][0])

                        else:
                            # if None in item[1]:
                            #     if item[1][0] is None:
                            #         if row[item[0]] >= int(item[1][1]):
                            #             full.append([rid, row['Trace']])
                            #             if var in leftRule:
                            #                 leftRule = re.sub(var + '<' + str(item[1][1]) + '.0', 'F', leftRule)
                            #             else:
                            #                 rightRule = re.sub(var + '<' + str(item[1][1]) + '.0', 'F', rightRule)
                            #
                            #             attribute = item[0]
                            #             activity = self.findActivity(rule, attribute)
                            #
                            #             dict = self.outputNLG(rid, nlp, activity, attribute, item[1][2], None, item[1][1], dict, row['Trace'])
                            #
                            #         else:
                            #             if var in leftRule:
                            #                 leftRule = re.sub(var + '<' + str(item[1][1]) + '.0', 'T', leftRule)
                            #             else:
                            #                 rightRule = re.sub(var + '<' + str(item[1][1]) + '.0', 'T', rightRule)
                            #     else:
                            #         if row[item[0]] <= int(item[1][0]):
                            #             full.append([rid, row['Trace']])
                            #
                            #             if var in leftRule:
                            #                 leftRule = re.sub(str(item[1][0]) + '.0<' + var, 'F', leftRule)
                            #             else:
                            #                 rightRule = re.sub(str(item[1][0]) + '.0<' + var, 'F', rightRule)
                            #
                            #
                            #             attribute = item[0]
                            #             activity = self.findActivity(rule, attribute)
                            #
                            #             dict = self.outputNLG(rid, nlp, activity, attribute, item[1][2], item[1][0], None, dict, row['Trace'])
                            #
                            #         else:
                            #             if var in leftRule:
                            #                 leftRule = re.sub(str(item[1][0]) + '.0<' + var, 'T', leftRule)
                            #             else:
                            #                 rightRule = re.sub(str(item[1][0]) + '.0<' + var, 'T', rightRule)
                            if None in item[1]:
                                if item[1][0] is None:
                                    if row[item[0]] >= int(item[1][1]):
                                        full.append([rid, row['Trace']])
                                        if var in leftRule:
                                            leftRule = self.replaceSubruleInRule(leftRule, item, var, 'F')
                                        else:
                                            rightRule = self.replaceSubruleInRule(rightRule, item, var, 'F')

                                        attribute = item[0]
                                        activity = self.findActivity(rule, attribute)

                                        dict = self.outputNLG(rid, nlp, activity, attribute, item[1][2], None,
                                                              item[1][1], dict, row['Trace'])

                                    else:
                                        if var in leftRule:
                                            leftRule = self.replaceSubruleInRule(leftRule, item, var, 'T')
                                        else:
                                            rightRule = self.replaceSubruleInRule(rightRule, item, var, 'T')

                                else:
                                    if row[item[0]] <= int(item[1][0]):
                                        full.append([rid, row['Trace']])

                                        if var in leftRule:
                                            leftRule = self.replaceSubruleInRule(leftRule, item, var, 'F')
                                        else:
                                            rightRule = self.replaceSubruleInRule(rightRule, item, var, 'F')

                                        attribute = item[0]
                                        activity = self.findActivity(rule, attribute)

                                        dict = self.outputNLG(rid, nlp, activity, attribute, item[1][2], item[1][0],
                                                              None, dict, row['Trace'])

                                    else:
                                        if var in leftRule:
                                            leftRule = self.replaceSubruleInRule(leftRule, item, var, 'T')
                                        else:
                                            rightRule = self.replaceSubruleInRule(rightRule, item, var, 'T')


                            else:
                                if row[item[0]] <= int(item[1][0]) or row[item[0]] >= int(item[1][1]):
                                    full.append([rid, row['Trace']])

                                    attribute = item[0]
                                    activity = self.findActivity(rule, attribute)
                                    if var in leftRule:
                                        leftRule = re.sub(str(item[1][0]) + '.0<' + var + '<' + str(item[1][1]) + '.0', 'F', leftRule)
                                    else:
                                        rightRule = re.sub(str(item[1][0]) + '.0<' + var + '<' + str(item[1][1]) + '.0', 'F', rightRule)

                                    dict = self.outputNLG(rid, nlp, activity, attribute, item[1][2], item[1][0], item[1][1], dict, row['Trace'])

                                else:
                                    if var in leftRule:
                                        leftRule = re.sub(str(item[1][0]) + '.0<' + var + '<' + str(item[1][1]) + '.0', 'T', leftRule)
                                    else:
                                        rightRule = re.sub(str(item[1][0]) + '.0<' + var + '<' + str(item[1][1]) + '.0', 'T', rightRule)
                    else:
                        breakItemNotInFrame = False
                        break

                if breakItemNotInFrame:
                    leftRule = re.sub('\(\s*', '(', leftRule)
                    rightRule = re.sub('\(\s*', '(', rightRule)
                    leftRule = re.sub('\s*\)', ')', leftRule)
                    rightRule = re.sub('\s*\)', ')',  rightRule)

                    if self.solveStringExpression(leftRule) and self.solveStringExpression(rightRule):
                        breakIt = False
                        if row['Trace'] in dict.keys():
                            for ruleName in dict[row['Trace']]:
                                if rid == ruleName:
                                    if len(dict[row['Trace']][rid]) == 1:
                                        breakIt = True
                                        break
                                    else:
                                        del dict[row['Trace']][rid][len(dict[row['Trace']][rid]) - 1]

                            if breakIt:
                                del dict[row['Trace']][rid]
                else:
                    break

        return dict

    def deviantTracesSplitTrees(self, positiveRules, nlp):
        merged_ = pd.read_csv(os.path.abspath('feature_vectors/csv_feature_vectors/negative/traces.csv'), index_col=0)
        positiveRules = pd.read_csv(
            os.path.abspath('redescription_mining/results/' + positiveRules + '-positive.queries'))

        merged = merged_.loc[rd.choices(list(merged_.index.values), k=7)]
        mergedColumns = ','.join(list(merged.columns))
        full = []
        dict = {}
        for rule in positiveRules.iterrows():
            rule = rule[1]
            rid = rule['rid']
            leftRule_ = re.sub('\|', 'or', rule['query_activation'])
            leftRule_ = re.sub('\&', 'and', leftRule_)

            rightRule_ = re.sub('\|', 'or', rule['query_target'])
            rightRule_ = re.sub('\&', 'and', rightRule_)

            if '_x' in mergedColumns and '_y' in mergedColumns:
                vars = re.findall(r'([A-Za-z]+)_', mergedColumns, re.S | re.I)
                if vars:
                    vars = set(vars)
                    leftQuery = rule['query_activation']
                    leftVars = rule['activation_vars']

                    rightQuery = rule['query_target']
                    rightVars = rule['target_vars']
                    for var in vars:
                        leftQuery = re.sub(var, var + '_x', leftQuery)
                        leftVars = re.sub(var, var + '_x', leftVars)

                        rightQuery = re.sub(var, var + '_y', rightQuery)
                        rightVars = re.sub(var, var + '_y', rightVars)

                    leftRule_ = re.sub('\|', 'or', leftQuery)
                    leftRule_ = re.sub('\&', 'and', leftRule_)

                    rightRule_ = re.sub('\|', 'or', rightQuery)
                    rightRule_ = re.sub('\&', 'and', rightRule_)

                    attributesInRule = self.extractSplitTrees(leftQuery, leftVars.split(',')) + self.extractSplitTrees(rightQuery,
                                                                                                   rightVars.split(','))


            else:

                attributesInRule = self.extractSplitTrees(rule['query_activation'],
                                                          rule['activation_vars'].split(',')) + self.extractSplitTrees(
                    rule['query_target'], rule['target_vars'].split(','))

            for row in merged.iterrows():
                row = row[1]
                leftRule = leftRule_
                rightRule = rightRule_

                breakItemNotInFrame = True
                for item in attributesInRule:
                    if item[0] in mergedColumns:
                        var = item[0]
                        if len(item[1]) == 1:
                            if item[1][0][1] == False:
                                if row[item[0]] != item[1][0][0]:
                                    full.append([rid, row['Trace']])

                                    attribute = item[0]
                                    activity = self.findActivity(rule, attribute)

                                    dict = self.outputNLG(rid, nlp, activity, attribute, item[1][0][2], item[1][0][0],
                                                          None, dict, row['Trace'])

                                    # if var in leftRule:
                                    #     leftRule = re.sub(var + '=' + str(item[1][0][0]), 'F', leftRule)
                                    # else:
                                    #     rightRule = re.sub(var + '=' + str(item[1][0][0]), 'F', rightRule)

                                    (leftRule, rightRule) = self.dealWithEqualnessAndBoolean(item[1][0][1], True, var,
                                                                                             leftRule, rightRule,
                                                                                             item[1][0][0])
                                else:
                                    # if var in leftRule:
                                    #     leftRule = re.sub(var + '=' + str(item[1][0][0]), 'T', leftRule)
                                    # else:
                                    #     rightRule = re.sub(var + '=' + str(item[1][0][0]), 'T', rightRule)
                                    (leftRule, rightRule) = self.dealWithEqualnessAndBoolean(item[1][0][1], False, var,
                                                                                             leftRule, rightRule,
                                                                                             item[1][0][0])

                            else:
                                if row[item[0]] == item[1][0][0]:

                                    full.append([rid, row['Trace']])
                                    attribute = item[0]
                                    activity = self.findActivity(rule, attribute)

                                    dict = self.outputNLG(rid, nlp, activity, attribute, item[1][0][2], item[1][0][0],
                                                          None, dict, row['Trace'])

                                    # if var in leftRule:
                                    #     leftRule = re.sub('!\s*' + var + '=' + str(item[1][0][0]), 'F', leftRule)
                                    # else:
                                    #     rightRule = re.sub('!\s*' + var + '=' + str(item[1][0][0]), 'F', rightRule)

                                    (leftRule, rightRule) = self.dealWithEqualnessAndBoolean(item[1][0][1], False, var,
                                                                                             leftRule, rightRule,
                                                                                             item[1][0][0])

                                else:
                                    # if var in leftRule:
                                    #     leftRule = re.sub('!\s*' + var + '=' + str(item[1][0][0]), 'T', leftRule)
                                    # else:
                                    #     rightRule = re.sub('!\s*' + var + '=' + str(item[1][0][0]), 'T', rightRule)
                                    (leftRule, rightRule) = self.dealWithEqualnessAndBoolean(item[1][0][1], True, var,
                                                                                             leftRule, rightRule,
                                                                                             item[1][0][0])

                        else:
                            if None in item[1]:
                                if item[1][0] is None:
                                    if row[item[0]] >= int(item[1][1]):
                                        full.append([rid, row['Trace']])
                                        if var in leftRule:
                                            leftRule = self.replaceSubruleInRule(leftRule, item, var, 'F')
                                        else:
                                            rightRule = self.replaceSubruleInRule(rightRule, item, var, 'F')

                                        attribute = item[0]
                                        activity = self.findActivity(rule, attribute)

                                        dict = self.outputNLG(rid, nlp, activity, attribute, item[1][2], None,
                                                              item[1][1], dict, row['Trace'])

                                    else:
                                        if var in leftRule:
                                            leftRule = self.replaceSubruleInRule(leftRule, item, var, 'T')
                                        else:
                                            rightRule = self.replaceSubruleInRule(rightRule, item, var, 'T')

                                else:
                                    if row[item[0]] <= int(item[1][0]):
                                        full.append([rid, row['Trace']])

                                        if var in leftRule:
                                            leftRule = self.replaceSubruleInRule(leftRule, item, var, 'F')
                                        else:
                                            rightRule = self.replaceSubruleInRule(rightRule, item, var, 'F')

                                        attribute = item[0]
                                        activity = self.findActivity(rule, attribute)

                                        dict = self.outputNLG(rid, nlp, activity, attribute, item[1][2], item[1][0],
                                                              None, dict, row['Trace'])

                                    else:
                                        if var in leftRule:
                                            leftRule = self.replaceSubruleInRule(leftRule, item, var, 'T')
                                        else:
                                            rightRule = self.replaceSubruleInRule(rightRule, item, var, 'T')


                            else:
                                if row[item[0]] <= int(item[1][0]) or row[item[0]] >= int(item[1][1]):
                                    full.append([rid, row['Trace']])

                                    attribute = item[0]
                                    activity = self.findActivity(rule, attribute)
                                    if var in leftRule:
                                        leftRule = re.sub(str(item[1][0]) + '.0<' + var + '<' + str(item[1][1]) + '.0',
                                                          'F', leftRule)
                                    else:
                                        rightRule = re.sub(str(item[1][0]) + '.0<' + var + '<' + str(item[1][1]) + '.0',
                                                           'F', rightRule)

                                    dict = self.outputNLG(rid, nlp, activity, attribute, item[1][2], item[1][0],
                                                          item[1][1], dict, row['Trace'])

                                else:
                                    if var in leftRule:
                                        leftRule = re.sub(str(item[1][0]) + '.0<' + var + '<' + str(item[1][1]) + '.0',
                                                          'T', leftRule)
                                    else:
                                        rightRule = re.sub(str(item[1][0]) + '.0<' + var + '<' + str(item[1][1]) + '.0',
                                                           'T', rightRule)
                    else:
                        breakItemNotInFrame = False
                        break

                if breakItemNotInFrame:
                    leftRule = re.sub('\(\s*', '(', leftRule)
                    rightRule = re.sub('\(\s*', '(', rightRule)
                    leftRule = re.sub('\s*\)', ')', leftRule)
                    rightRule = re.sub('\s*\)', ')', rightRule)
                    leftRule = re.sub('\! ', '', leftRule)
                    rightRule = re.sub('\! ', '', rightRule)

                    if self.solveStringExpression(leftRule) and self.solveStringExpression(rightRule):
                        breakIt = False
                        if row['Trace'] in dict.keys():
                            for ruleName in dict[row['Trace']]:
                                if rid == ruleName:
                                    if len(dict[row['Trace']][rid]) == 1:
                                        breakIt = True
                                        break
                                    else:
                                        del dict[row['Trace']][rid][len(dict[row['Trace']][rid]) - 1]

                            if breakIt:
                                del dict[row['Trace']][rid]
                else:
                    break

        return dict

    def replaceSubruleInRule(self, rule, item, var, keepF):
        if item[1][0] is None:
            if keepF == 'F':
                if item[1][3] is True:
                    rule = re.sub(str(item[1][1]) + '.0<' + var, 'F', rule)
                else:
                    rule = re.sub(var + '<' + str(item[1][1]) + '.0', 'F', rule)
            else:
                if item[1][3] is True:
                    rule = re.sub(str(item[1][1]) + '.0<' + var, 'T', rule)
                else:
                    rule = re.sub(var + '<' + str(item[1][1]) + '.0', 'T', rule)

        else:
            if keepF == 'F':
                if item[1][3] is True:
                    rule = re.sub(var + '<' + str(item[1][0]) + '.0', 'F', rule)
                else:
                    rule = re.sub(str(item[1][0]) + '.0<' + var, 'F', rule)
            else:
                if item[1][3] is True:
                    rule = re.sub(var + '<' + str(item[1][0]) + '.0', 'T', rule)
                else:
                    rule = re.sub(str(item[1][0]) + '.0<' + var, 'T', rule)

        return rule

    def dealWithEqualnessAndBoolean(self, negation, trueOrFalse, var, leftRule, rightRule, value):
        if negation == False:
            if trueOrFalse:
                if (var +'=') in leftRule or (var +'=') in rightRule:
                    if var in leftRule:
                        leftRule = re.sub(var + '=' + str(value), 'F', leftRule)
                    else:
                        rightRule = re.sub(var + '=' + str(value), 'F', rightRule)

                else:
                    if var in leftRule:
                        leftRule = re.sub(var, 'F', leftRule)
                    else:
                        rightRule = re.sub(var, 'F', rightRule)
            else:
                if (var + '=') in leftRule or (var + '=') in rightRule:
                    if var in leftRule:
                        leftRule = re.sub(var + '=' + str(value), 'T', leftRule)
                    else:
                        rightRule = re.sub(var + '=' + str(value), 'T', rightRule)

                else:
                    if var in leftRule:
                        leftRule = re.sub(var, 'T', leftRule)
                    else:
                        rightRule = re.sub(var, 'T', rightRule)
        else:
            if trueOrFalse:
                if (var + '=') in leftRule or (var + '=') in rightRule:
                    if var in leftRule:
                        leftRule = re.sub(var + '=' + str(value), 'T', leftRule)
                    else:
                        rightRule = re.sub(var + '=' + str(value), 'T', rightRule)

                else:
                    if var in leftRule:
                        leftRule = re.sub(var, 'F', leftRule)
                    else:
                        rightRule = re.sub(var, 'F', rightRule)

            else:
                if (var + '=') in leftRule or (var + '=') in rightRule:
                    if var in leftRule:
                        leftRule = re.sub(var + '=' + str(value), 'F', leftRule)
                    else:
                        rightRule = re.sub(var + '=' + str(value), 'F', rightRule)

                else:
                    if var in leftRule:
                        leftRule = re.sub(var, 'T', leftRule)
                    else:
                        rightRule = re.sub(var, 'T', rightRule)

        return (leftRule, rightRule)

    def solveRuleForRow(self, rule, row, attRules):
        vars = list(rule['activation_vars'].split(',')) + list(rule['target_vars'].split(','))
        leftRule = re.sub('\|', 'or', rule['query_activation'])
        leftRule = re.sub('\&', 'and', leftRule)

        rightRule = re.sub('\|', 'or', rule['query_target'])
        rightRule = re.sub('\&', 'and', rightRule)

        for var in vars:
            for item in attRules:
                if var == item[0]:
                    if len(item[1]) > 1:
                        item = item[1]
                    else:
                        item = item[1][0]

                    type = item[2]
                    if type == '<<':
                        if int(item[0]) < row[var] and row[var] < int(item[1]):
                            if var in leftRule:
                                leftRule = re.sub(str(item[0]) +'.0<'+var + '<'+str(item[1])+'.0', 'T', leftRule)
                            else:
                                rightRule = re.sub(str(item[0]) +'.0<'+var + '<'+str(item[1])+'.0', 'T', rightRule)
                        else:
                            if var in leftRule:
                                leftRule = re.sub(str(item[0]) +'.0<'+var + '<'+str(item[1])+'.0', 'F', leftRule)
                            else:
                                rightRule = re.sub(str(item[0]) +'.0<'+var + '<'+str(item[1])+'.0', 'F', rightRule)

                    elif type == '1<':
                        if int(item[0]) < row[var]:
                            if var in leftRule:
                                leftRule = re.sub(str(item[0]) +'.0<'+var, 'T', leftRule)
                            else:
                                rightRule = re.sub(str(item[0]) +'.0<'+var, 'T', rightRule)
                        else:
                            if var in leftRule:
                                leftRule = re.sub(str(item[0]) +'.0<'+var , 'F', leftRule)
                            else:
                                rightRule = re.sub(str(item[0]) +'.0<'+var, 'F', rightRule)

                    elif type == '2<':
                        if row[var] < int(item[1]):
                            if var in leftRule:
                                leftRule = re.sub( var + '<' + str(item[1]) + '.0', 'T', leftRule)
                            else:
                                rightRule = re.sub(var + '<' + str(item[1]) + '.0', 'T',
                                                   rightRule)
                        else:
                            if var in leftRule:
                                leftRule = re.sub(var + '<' + str(item[1]) + '.0', 'F', leftRule)
                            else:
                                rightRule = re.sub(var + '<' + str(item[1]) + '.0', 'F',
                                                   rightRule)
                    elif type == '=':
                        # if isinstance(item[0], int):
                        #     if int(item[0]) == row[var]:
                        #         if var in leftRule:
                        #             leftRule = re.sub(var + '.0=' + str(item[0]), 'T', leftRule)
                        #         else:
                        #             rightRule = re.sub(var + '.0=' + str(item[0]), 'T', rightRule)
                        #     else:
                        #         if var in leftRule:
                        #             leftRule = re.sub(var + '.0=' + str(item[0]), 'F', leftRule)
                        #         else:
                        #             rightRule = re.sub(var + '.0=' + str(item[0]), 'F', rightRule)
                        # else:
                        if item[1]:
                            if item[0] == row[var]:
                                if var in leftRule:
                                    leftRule = re.sub('!\s*'+ var + '=' + str(item[0]), 'F', leftRule)
                                else:
                                    rightRule = re.sub('!\s*'+ var + '=' + str(item[0]), 'F', rightRule)
                            else:
                                if var in leftRule:
                                    leftRule = re.sub('!\s*'+ var + '=' + str(item[0]), 'T', leftRule)
                                else:
                                    rightRule = re.sub('!\s*'+ var + '=' + str(item[0]), 'T', rightRule)
                        else:
                            if item[0] == row[var]:
                                if var in leftRule:
                                    leftRule = re.sub(var + '=' + str(item[0]), 'T', leftRule)
                                else:
                                    rightRule = re.sub(var + '=' + str(item[0]), 'T', rightRule)
                            else:
                                if var in leftRule:
                                    leftRule = re.sub(var + '=' + str(item[0]), 'F', leftRule)
                                else:
                                    rightRule = re.sub(var + '=' + str(item[0]), 'F', rightRule)

        leftRule = re.sub('!\s*', '', leftRule)
        rightRule = re.sub('!\s*', '', rightRule)

        return not (self.solveStringExpression(leftRule) and self.solveStringExpression(rightRule))

    def solveRuleForFrame(self, rule, frame, attRules):
        vars = list(rule['activation_vars'].split(',')) + list(rule['target_vars'].split(','))
        leftRule = re.sub('\|', 'or', rule['query_activation'])
        leftRule = re.sub('\&', 'and', leftRule)

        rightRule = re.sub('\|', 'or', rule['query_target'])
        rightRule = re.sub('\&', 'and', rightRule)

        newFrame = pd.DataFrame(columns=frame.columns)
        index = 0
        for row in frame.iterrows():
            row = row[1]
            for var in vars:
                for item in attRules:
                    if var == item[0]:
                        if len(item[1]) > 1:
                            item = item[1]
                        else:
                            item = item[1][0]

                        type = item[2]
                        if type == '<<':
                            if int(item[0]) < row[var] and row[var] < int(item[1]):
                                if var in leftRule:
                                    leftRule = re.sub(str(item[0]) + '.0<' + var + '<' + str(item[1]) + '.0', 'T', leftRule)
                                else:
                                    rightRule = re.sub(str(item[0]) + '.0<' + var + '<' + str(item[1]) + '.0', 'T',
                                                       rightRule)
                            else:
                                if var in leftRule:
                                    leftRule = re.sub(str(item[0]) + '.0<' + var + '<' + str(item[1]) + '.0', 'F', leftRule)
                                else:
                                    rightRule = re.sub(str(item[0]) + '.0<' + var + '<' + str(item[1]) + '.0', 'F',
                                                       rightRule)

                        elif type == '1<':
                            if int(item[0]) < row[var]:
                                if var in leftRule:
                                    leftRule = re.sub(str(item[0]) + '.0<' + var, 'T', leftRule)
                                else:
                                    rightRule = re.sub(str(item[0]) + '.0<' + var, 'T', rightRule)
                            else:
                                if var in leftRule:
                                    leftRule = re.sub(str(item[0]) + '.0<' + var, 'F', leftRule)
                                else:
                                    rightRule = re.sub(str(item[0]) + '.0<' + var, 'F', rightRule)

                        elif type == '2<':
                            if row[var] < int(item[1]):
                                if var in leftRule:
                                    leftRule = re.sub(var + '<' + str(item[1]) + '.0', 'T', leftRule)
                                else:
                                    rightRule = re.sub(var + '<' + str(item[1]) + '.0', 'T',
                                                       rightRule)
                            else:
                                if var in leftRule:
                                    leftRule = re.sub(var + '<' + str(item[1]) + '.0', 'F', leftRule)
                                else:
                                    rightRule = re.sub(var + '<' + str(item[1]) + '.0', 'F',
                                                       rightRule)
                        elif type == '=':
                            # if isinstance(item[0], int):
                            #     if int(item[0]) == row[var]:
                            #         if var in leftRule:
                            #             leftRule = re.sub(var + '.0=' + str(item[0]), 'T', leftRule)
                            #         else:
                            #             rightRule = re.sub(var + '.0=' + str(item[0]), 'T', rightRule)
                            #     else:
                            #         if var in leftRule:
                            #             leftRule = re.sub(var + '.0=' + str(item[0]), 'F', leftRule)
                            #         else:
                            #             rightRule = re.sub(var + '.0=' + str(item[0]), 'F', rightRule)
                            # else:
                            if item[1]:
                                if item[0] == row[var]:
                                    if var in leftRule:
                                        leftRule = re.sub('!\s*' + var + '=' + str(item[0]), 'F', leftRule)
                                    else:
                                        rightRule = re.sub('!\s*' + var + '=' + str(item[0]), 'F', rightRule)
                                else:
                                    if var in leftRule:
                                        leftRule = re.sub('!\s*' + var + '=' + str(item[0]), 'T', leftRule)
                                    else:
                                        rightRule = re.sub('!\s*' + var + '=' + str(item[0]), 'T', rightRule)
                            else:
                                if item[0] == row[var]:
                                    if var in leftRule:
                                        leftRule = re.sub(var + '=' + str(item[0]), 'T', leftRule)
                                    else:
                                        rightRule = re.sub(var + '=' + str(item[0]), 'T', rightRule)
                                else:
                                    if var in leftRule:
                                        leftRule = re.sub(var + '=' + str(item[0]), 'F', leftRule)
                                    else:
                                        rightRule = re.sub(var + '=' + str(item[0]), 'F', rightRule)

            leftRule = re.sub('!\s*', '', leftRule)
            rightRule = re.sub('!\s*', '', rightRule)

            if not (self.solveStringExpression(leftRule) and self.solveStringExpression(rightRule)):
                newFrame.loc[index] = row
                index +=1

        return newFrame

    def extract(self, rule, vars):
        dict = []
        for attribute in vars:
            if re.search(r'([0-9.]+)[<>]+' + attribute + '[<>]+([0-9.]+)', rule, re.S | re.I):
                first = re.search(r'([0-9.]+)[<>]+' + attribute + '[<>]+([0-9.]+)', rule, re.S | re.I).group(1)
                second = re.search(r'([0-9.]+)[<>]+' + attribute + '[<>]+([0-9.]+)', rule, re.S | re.I).group(2)
                first = re.sub(r'\.[0]+', '', first)
                second = re.sub(r'\.[0]+', '', second)

                dict.append((attribute, [first, second, '<<']))

            elif re.search(r'([0-9.]+)[<>]+' + attribute, rule, re.S | re.I):
                first = re.search(r'([0-9.]+)[<>]+' + attribute, rule, re.S | re.I).group(1)
                first = re.sub(r'\.[0]+', '', first)

                if re.search(r'!\s*([0-9.]+)[<>]+' + attribute, rule, re.S | re.I):
                    dict.append((attribute, [None, first, '2<', True]))
                else:
                    dict.append((attribute, [first, None, '1<', False]))



            elif re.search(attribute + '[<>]+([0-9.]+)', rule, re.S | re.I):
                second = re.search(attribute + '[<>]+([0-9.]+)', rule, re.S | re.I).group(1)
                second = re.sub(r'\.[0]+', '', second)

                if re.search('!\s*'+ attribute + '[<>]+([0-9.]+)', rule, re.S | re.I):
                    dict.append((attribute, [second, None, '1<', True]))
                else:
                    dict.append((attribute, [None, second, '2<', False]))


            elif re.search(attribute + '=([A-Za-z0-9.]*)', rule, re.S | re.I):
                first = re.search(attribute + '=([A-Za-z0-9.]*)', rule, re.S | re.I).group(1)
                first = re.sub(r'\.[0]+', '', first)

                if re.search('!\s*'+attribute + '=([A-Za-z0-9.]*)', rule, re.S | re.I):
                    dict.append((attribute, [(first, True, '=')]))
                else:
                    dict.append((attribute, [(first,False, '=')]))

            elif re.search(attribute, rule, re.S | re.I):
                if re.search('!\s*' + attribute, rule, re.S | re.I):
                    dict.append((attribute, [('False', True, '=')]))
                else:
                    dict.append((attribute, [('True', False, '=')]))

        return dict

    def extractSplitTrees(self, rule, vars):
        dict = []
        for attribute in vars:
            for first, last in re.findall(r'([0-9.]+)[<>]+' + attribute + '[<>]+([0-9.]+)', rule, re.S | re.I):
                rule = re.sub(first+'[<>]+' + attribute + second, '', rule)
                first = re.sub(r'\.[0]+', '', first)
                second = re.sub(r'\.[0]+', '', second)

                dict.append((attribute, [first, second, '<<']))

            for first in re.findall(r'([0-9.]+)[<>]+' + attribute, rule, re.S | re.I):
                if re.search(r'!\s*' + first + '[<>]+' + attribute, rule, re.S | re.I):
                    rule = re.sub(r'!\s*' + first + '[<>]+' + attribute, '', rule)
                    first = re.sub(r'\.[0]+', '', first)
                    dict.append((attribute, [None, first, '2<', True]))
                else:
                    rule = re.sub(first + '[<>]+' + attribute, '', rule)

                    first = re.sub(r'\.[0]+', '', first)
                    dict.append((attribute, [first, None, '1<', False]))

            for second in re.findall(attribute + '[<>]+([0-9.]+)', rule, re.S | re.I):

                if re.search('!\s*' + attribute + '[<>]+' + second, rule, re.S | re.I):
                    rule = re.sub('!\s*' + attribute + '[<>]+' + second, '', rule)
                    second = re.sub(r'\.[0]+', '', second)

                    dict.append((attribute, [second, None, '1<', True]))
                else:
                    rule = re.sub(attribute + '[<>]+' + second, '', rule)
                    second = re.sub(r'\.[0]+', '', second)
                    dict.append((attribute, [None, second, '2<', False]))

            for first in re.findall(attribute + '=([A-Za-z0-9.]*)', rule, re.S | re.I):

                if re.search('!\s*' + attribute + '=' + first, rule, re.S | re.I):
                    rule = re.sub(r'!\s*' + attribute + '=' + first, '', rule)
                    first = re.sub(r'\.[0]+', '', first)

                    dict.append((attribute, [(first, True, '=')]))
                else:
                    rule = re.sub(attribute + '=' + first, '', rule)
                    dict.append((attribute, [(first, False, '=')]))

            # for item in re.search(attribute, rule, re.S | re.I):
            #     if re.search('!\s*' + attribute, rule, re.S | re.I):
            #         dict.append((attribute, [('False', True, '=')]))
            #     else:
            #         dict.append((attribute, [('True', False, '=')]))

        return dict

    def removeDup(self, listt, tuple):
        if tuple not in listt:
            listt.append(tuple)
        return listt

    def representationOfNegativeContraint(self, nlp, activity, attribute, type, first, second):
        if activity in attribute:
            attribute = re.sub(activity, '', attribute)

        attribute = re.sub(r"(\w)([A-Z])", r"\1 \2", attribute)

        activity = re.sub(r"(\w)([A-Z])", r"\1 \2", activity)
        doc = nlp(activity)
        tokens = doc.sentences[0].tokens

        for index, item in enumerate(tokens):
            item = item.words[0]

            if item.xpos == 'NN' or item.xpos == 'NNP':
                activityExtracted = item.text

        if activityExtracted in attribute or attribute in activityExtracted:
            attribute = re.sub(activityExtracted, '', attribute).strip()

        if type == '<<' and first != second:
            listToChoose = ['the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' does not vary between ' + str(
                first) + ' and ' + str(second) + ' ',
                            'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' does not range from ' + str(
                                first) + ' to ' + str(second) + ' ',
                            'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' doesn\'t alternate between ' + str(
                                first) + ' and ' + str(second) + ' ',
                            'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' doesn\'t stretch from ' + str(
                                first) + ' to ' + str(second) + ' ',
                            ]

            return rd.choice(listToChoose)

        elif type == '1<':

            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not bigger than ' + str(first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is smaller than ' + str(first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' doesn\'t exceed ' + str(
                    first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' goes lower than ' + str(
                    first) + ' '
                ]

            return rd.choice(listToChoose)

        elif type == '2<':
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not below ' + str(second) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not smaller than ' + str(second) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is above ' + str(
                    second) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is higher than ' + str(
                    second) + ' '
            ]

            return rd.choice(listToChoose)

        elif type == '=':
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is unequal to ' + str(first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not ' + str(first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' differs from ' + str(first) + ' '
            ]
            if attribute.lower() == 'resource':
                listToChoose = [
                    'the process is not executed by ' + str(
                        first) + ' ',
                    # 'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not ' + str(first) + ' ',
                    # 'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' differs from ' + str(first) + ' '
                ]

            return rd.choice(listToChoose)

        elif type == '<<' and first == second:
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is unequal to ' + str(first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not ' + str(first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' differs from ' + str(first) + ' '
            ]

            return rd.choice(listToChoose)

        elif self.type == 'boolean':
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is equal to ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is ' + str(self.first) + ' '

            ]

            return rd.choice(listToChoose)

    def findActivity(self, rule, attribute):
        if '_x' in attribute or '_y' in attribute:
            attribute = re.sub('_[a-z]', '', attribute)

        if attribute in rule['activation_vars']:
            return rule['activation_activity']

        elif attribute in rule['target_vars']:
            return rule['target_activity']

    def outputNLG(self, ruleID, nlp, activity, attribute, type, first, second, dict=None, trace=None):
        attribute = re.sub('_[a-z]', '', attribute)
        if self.dictRule == {}:
            self.dictRule[ruleID] = [trace]
        else:
            if ruleID in self.dictRule.keys():
                self.dictRule[ruleID].append(trace)
            else:
                self.dictRule[ruleID] = [trace]

        if dict is None:
            dict = {}
            dict[ruleID] = self.representationOfNegativeContraint(nlp, activity, attribute, type, first, second)

            return dict
        else:
            if trace in dict.keys():
                if ruleID in dict[trace].keys():
                    dict[trace][ruleID].append(self.representationOfNegativeContraint(nlp, activity, attribute, type, first, second))
                else:
                    dict[trace][ruleID] = [self.representationOfNegativeContraint(nlp, activity, attribute, type, first, second)]
            else:
                dict[trace] = {ruleID : [self.representationOfNegativeContraint(nlp, activity, attribute, type, first, second)] }

            return dict

    def solveStringExpression(self, s):
        stack = []
        op = {
            "or": lambda x, y: x or y,
            "and": lambda x, y: x and y,
        }
        for v in s.split():
            if v[0] == "(":
                stack.append(v[v.count("("):] == "T")
            elif v.count(")") > 0:
                ct = v.count(")")
                stack.append(v[:-ct] == "T")
                for _ in range(ct):
                    right = stack.pop()
                    o = stack.pop()
                    left = stack.pop()
                    stack.append(o(left, right))
            elif v in ["T", "F"]:
                stack.append(v == "T")
            else:
                stack.append(op[v])

        if len(stack) > 1:
            for i in range(0, len(stack) - 1, 2):
                stack[i + 2] = stack[i + 1](stack[i], stack[i + 2])
            return stack[-1]

        return stack[0]
