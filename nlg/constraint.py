# @project Deviance Analysis by Means of Redescription Mining - Master Thesis
# @author Engjëll Ahmeti
# @date 1/20/2021
import re
import pandas as pd
import os
from nlg.dsynts import DSyntS
from nlg.extraction import ExtractionConstraint
import random as rd

class Constraint:
    def __init__(self, attribute, first, second, type, activity, nlp, negation=False, extractDSyntSOnLeafs=True, activationOrTarget=''):
        if first == '':
            first = None
        if second == '':
            second = None

        self.activationOrTarget = activationOrTarget
        self.negation = negation
        self.second = None
        self.nlp = nlp
        self.activity = activity
        self.attribute = attribute
        if type == '<' or type == '>':
            self.first = first
            self.second = second

            if self.first and self.second:
                self.first = re.sub('\.0', '', first)
                self.second = re.sub('\.0', '', second)
                self.interval = '(' + self.first + ', ' + self.second + ')'
                self.type = '<<'

            elif self.first:
                self.first = re.sub('\.0', '', first)
                self.interval = '(' + self.first + ', +∞)'
                self.type = '1<'

            elif self.second:
                self.second = re.sub('\.0', '', second)
                self.interval = '(-∞, ' + self.second + ')'
                self.type = '2<'


        elif type == '=':
            self.first = re.sub('\".0', '', first)
            self.interval = first
            self.type = '='

        elif type == 'boolean':
            if self.negation:
                self.first = 'False'
            else:
                self.first = 'True'

            self.interval = self.first
            self.type = 'boolean'

        if self.negation:
            self.output = self.representationOfNegativeContraint()
        else:
            self.output = self.representationOfContraint()

        extract = ExtractionConstraint(self.output, self.nlp, self.attribute, extractDSyntSOnLeafs, self.activity, self.interval, self.activationOrTarget, self.type, self.first, self.second)

        self.tree = extract.tree
        self.SPL = extract.SPL
        self.CoNLL = extract.CoNLL
        self.DSyntS = extract.DSyntS
        # self.tree = self.extractTree()
        # self.CoNLL = self.extractCoNLL()
        # self.DSyntS = self.extractDSyntS()

    def __str__(self) -> str:
        return self.output

    def representationOfContraint(self):
        attribute = self.attribute
        if self.activity in self.attribute:
            attribute = re.sub(self.activity, '', self.attribute)
        attribute = re.sub(r"(\w)([A-Z])", r"\1 \2", attribute)

        activity = re.sub(r"(\w)([A-Z])", r"\1 \2", self.activity)
        doc = self.nlp(activity)
        tokens = doc.sentences[0].tokens

        for index, item in enumerate(tokens):
            item = item.words[0]

            if item.xpos == 'NN' or item.xpos == 'NNP':
                activityExtracted = item.text

        if activityExtracted in attribute or attribute in activityExtracted:
            attribute = re.sub(activityExtracted, '', attribute).strip()

        if self.type == '<<' and self.first != self.second:
            listToChoose = ['the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' varies between ' + str(self.first) + ' and ' + str(self.second) + ' ',
                            'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' ranges from ' + str(
                                self.first) + ' to ' + str(self.second) + ' ',
                            'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' alternates between ' + str(
                                self.first) + ' and ' + str(self.second) + ' ',
                            'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' stretches from ' + str(
                                self.first) + ' to ' + str(self.second) + ' ',
                            ]

            return rd.choice(listToChoose)

        elif self.type == '1<':
            listToChoose = ['the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is bigger than ' + str(self.first) + ' ',
                            'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' exceeds ' + str(
                                self.first) + ' ',
                            'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' doesn\'t go lower than ' + str(
                                self.first) + ' '
            ]

            return rd.choice(listToChoose)

        elif self.type == '2<':
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is below ' + str(self.second) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is smaller than ' + str(self.second) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is lesser than ' + str(
                    self.second) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is lower than ' + str(
                    self.second) + ' '
                ]

            return rd.choice(listToChoose)

        elif self.type == '=':
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is equal to ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is ' + str(self.first) + ' ',
                # 'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' consists of ' + str(self.first) + ' ',

            ]
            if attribute.lower() == 'resource':
                listToChoose = [
                    'the process is executed by ' + str(
                        self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is ' + str(self.first) + ' ',
                ]

            return rd.choice(listToChoose)

        elif self.type == '<<' and self.first == self.second:
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is equal to ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' consists of ' + str(self.first) + ' ',

            ]

            return rd.choice(listToChoose)

        elif self.type == 'boolean':
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is equal to ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is ' + str(self.first) + ' '

            ]

            return rd.choice(listToChoose)

    def representationOfNegativeContraint(self):
        attribute = self.attribute
        if self.activity in self.attribute:
            attribute = re.sub(self.activity, '', self.attribute)

        attribute = re.sub(r"(\w)([A-Z])", r"\1 \2", attribute)

        activity = re.sub(r"(\w)([A-Z])", r"\1 \2", self.activity)
        doc = self.nlp(activity)
        tokens = doc.sentences[0].tokens

        for index, item in enumerate(tokens):
            item = item.words[0]

            if item.xpos == 'NN' or item.xpos == 'NNP':
                activityExtracted = item.text

        if activityExtracted in attribute or attribute in activityExtracted:
            attribute = re.sub(activityExtracted, '', attribute).strip()

        if self.type == '<<' and self.first != self.second:
            listToChoose = ['the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' does not vary between ' + str(
                self.first) + ' and ' + str(self.second) + ' ',
                            'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' does not range from ' + str(
                                self.first) + ' to ' + str(self.second) + ' ',
                            'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' doesn\'t alternate between ' + str(
                                self.first) + ' and ' + str(self.second) + ' ',
                            'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' doesn\'t stretch from ' + str(
                                self.first) + ' to ' + str(self.second) + ' ',
                            ]

            return rd.choice(listToChoose)

        elif self.type == '1<':

            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not bigger than ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is smaller than ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' doesn\'t exceed ' + str(
                    self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' goes lower than ' + str(
                    self.first) + ' '
                ]

            return rd.choice(listToChoose)

        elif self.type == '2<':
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not below ' + str(self.second) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not smaller than ' + str(self.second) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is above ' + str(
                    self.second) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is higher than ' + str(
                    self.second) + ' '
            ]

            return rd.choice(listToChoose)

        elif self.type == '=':
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is unequal to ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' differs from ' + str(self.first) + ' '
            ]
            if attribute.lower() == 'resource':
                listToChoose = [
                    'the process is not executed by ' + str(
                        self.first) + ' ',
                    'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not ' + str(self.first) + ' ',
                    'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' differs from ' + str(self.first) + ' '
                ]

            return rd.choice(listToChoose)

        elif self.type == '<<' and self.first == self.second:
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is unequal to ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is not ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' differs from ' + str(self.first) + ' '
            ]

            return rd.choice(listToChoose)

        elif self.type == 'boolean':
            listToChoose = [
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is equal to ' + str(self.first) + ' ',
                'the ' + activityExtracted.lower() + ' ' + attribute.lower() + ' is ' + str(self.first) + ' '

            ]

            return rd.choice(listToChoose)