# @project Deviance Analysis by Means of Redescription Mining - Master Thesis 
# @author Engjëll Ahmeti
# @date 1/31/2021
import os
import pandas as pd
from nlg.dsynts import DSyntS
import re

class ExtractionConstraint:
    def __init__(self, output, nlp, attribute, extractDSyntSOnLeafs, activity=None, interval=None, activationOrTarget='', type=None, first=None, second=None):
        self.nlp = nlp
        self.attribute = attribute
        self.activity = activity
        self.interval = interval
        self.output = output
        self.activationOrTarget = activationOrTarget
        self.type = type
        self.first = first
        self.second = second

        if extractDSyntSOnLeafs:
            self.CoNLL = self.extractCoNLL(output)
            self.DSyntS = self.extractDSyntS()
        else:
            self.CoNLL = None
            self.DSyntS = None

        if activity:
            self.tree = self.extractTree()
            self.SPL = self.extractSPL()

    def extractTree(self):
        dict = {}
        dict[self.attribute] = {'activity': self.activity, 'interval': self.interval, 'output':self.output, 'activationOrTarget':self.activationOrTarget}

        return dict

    def extractCoNLL(self, output):
        # nlp = stanza.Pipeline('en')

        doc = self.nlp(output)
        tokens = doc.sentences[0].tokens

        finalTokens = []
        for index, item in enumerate(tokens):
            item = item.words[0]

            if item.lemma == 'and':
                item.deprel = 'COORD'

            elif item.lemma == 'between':
                item.deprel = 'AMOD'

            elif item.deprel == 'compound' or item.xpos == 'CD':
                item.deprel = 'NMOD'

            finalTokens.append(item)

        dict = {}
        dict[self.attribute] = finalTokens

        return dict

    def extractDSyntS(self):
        conllPath = os.path.abspath("nlg/tool/tree.conll")
        dsyntsPath = os.path.abspath("nlg/tool/DSyntS/tree.conll/tree.conll_out.conll")

        with open(conllPath, 'wt') as a:
            a.write('')

        with open(dsyntsPath, 'wt') as a:
            a.write('')

        for index, item in enumerate(self.CoNLL[self.attribute]):
            keepList = []
            keepList.append(item.id)
            keepList.append(item.text)
            keepList.append(item.lemma)
            keepList.append(item.lemma)
            keepList.append(item.xpos)
            keepList.append(item.xpos)
            keepList.append('_')
            keepList.append('_')

            keepList.append(item.head)
            keepList.append(item.head)
            if item.lemma == 'and':
                keepList.append('COORD')
                keepList.append('COORD')
            elif item.lemma == 'between':
                keepList.append('AMOD')
                keepList.append('ADV')
            elif item.deprel == 'compound' or item.xpos == 'CD' or item.deprel == 'nsubj':
                keepList.append('NMOD')
                keepList.append('NMOD')
            else:
                keepList.append(item.deprel.upper())
                keepList.append(item.deprel.upper())

            keepList.append('_')
            keepList.append('_')
            keepList.append('_')
            keepList.append('_')
            keepList.append('_')
            keepList.append('_')

            output = '\t'.join([str(x) for x in keepList])
            with open(conllPath, 'at') as a:
                a.write(output + '\n')

        with open(conllPath, 'at') as a:
            a.write('\n')

        os.system('java -jar {0} -i {1} -o {2}'.format('nlg/tool/converter.jar', 'nlg/tool/tree.conll', 'nlg/tool/DSyntS'))

        dsyntsFrame = pd.read_csv(dsyntsPath, delimiter='\t', header=None)
        dsyntsFrame.columns = ['ID', 'TEXT', 'LEMMA', 'PLEMMA', 'POS', 'PPOS', 'FEAT', 'PFEAT', 'HEAD', 'PHEAD',
                               'DEPREL', 'PDEPREL', 'FILLPRED', 'PRED']

        listOfDsynts = []
        for row in dsyntsFrame.iterrows():
            row = row[1]
            listOfDsynts.append(DSyntS(row))

        dict = {}
        dict[self.attribute] = listOfDsynts

        return dict

    def extractSPL(self):
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

        spl = ''
        if self.type == '<<' and self.first != self.second:
            spl = r'(BETWEENRANGE / PROPERTY-ASCRIPTION  :LEX ALTERNATE  :DOMAIN  (ATTRIBUTENAME / SUBJECT :LEX {0} :DETERMINER THE) :RANGE (FROMTO / OBJECT :LEX IN :NUMBER MASS  :BETWEEN (IR / IR :EXPERIENTIAL-COMPLEXITY-Q COMPLEX :CONJUNCTIVE-EXTENSION-Q CONJUNCTIVE :COMPLEX-THING-PART1-ID (FIRST / ABSTRACTION :LEX {1}\, :NUMBER MASS)  :COMPLEX-THING-PART2-ID (SECOND / ABSTRACTION :LEX {2}\. :NUMBER MASS)) ) )'.format(
                attribute.upper(), self.first.upper(), self.second.upper())
            # spl = r'(BETWEENRANGE / PROPERTY-ASCRIPTION  :LEX ALTERNATE  :DOMAIN  (ATTRIBUTENAME / SUBJECT :LEX {0} :DETERMINER THE) :RANGE (FROMTO / OBJECT :LEX IN :NUMBER MASS  :BETWEEN (IR / IR :EXPERIENTIAL-COMPLEXITY-Q COMPLEX :CONJUNCTIVE-EXTENSION-Q CONJUNCTIVE :COMPLEX-THING-PART1-ID (FIRST / ABSTRACTION :QUANTITY {1})  :COMPLEX-THING-PART2-ID (SECOND / ABSTRACTION :QUANTITY {2})) ) )'.format(attribute.upper(), self.first.upper(), self.second.upper())

        elif self.type == '1<':
            spl = r'(HIGHERTHAN / GREATER-THAN-COMPARISON  :TENSE PRESENT  :DOMAIN (ATTRIBUTENAME / ONE-OR-TWO-D-TIME :LEX {0} :DETERMINER THE) :RANGE (COMPARISONTYPE / SENSE-AND-MEASURE-QUALITY :LEX HIGH) :STANDARD (FIRST / QUALITY :LEX {1}\. :DETERMINER ZERO) )'.format(
                attribute.upper(), self.first.upper())
            # spl = r'(HIGHERTHAN / GREATER-THAN-COMPARISON  :TENSE PRESENT  :DOMAIN (ATTRIBUTENAME / ONE-OR-TWO-D-TIME :LEX {0} :DETERMINER THE) :RANGE (COMPARISONTYPE / SENSE-AND-MEASURE-QUALITY :LEX HIGH) :STANDARD (FIRST / ABSTRACTION :QUANTITY {1}) )'.format(
            #     attribute.upper(), self.first.upper())
        elif self.type == '2<':
            spl = r'(LOWERTHAN / GREATER-THAN-COMPARISON  :TENSE PRESENT  :DOMAIN (ATTRIBUTENAME / ONE-OR-TWO-D-TIME :LEX {0} :DETERMINER THE) :RANGE (COMPARISONTYPE / SENSE-AND-MEASURE-QUALITY :LEX LOW) :STANDARD (SECOND / QUALITY :LEX {1}\. :DETERMINER ZERO) ) '.format(
                attribute.upper(), self.second.upper())
            # spl = r'(LOWERTHAN / GREATER-THAN-COMPARISON  :TENSE PRESENT  :DOMAIN (ATTRIBUTENAME / ONE-OR-TWO-D-TIME :LEX {0} :DETERMINER THE) :RANGE (COMPARISONTYPE / SENSE-AND-MEASURE-QUALITY :LEX LOW) :STANDARD (SECOND / ABSTRACTION :QUANTITY {1}) ) '.format(
            #     attribute.upper(), self.second.upper())
        elif self.type == '=':
            spl = r'(EQUALNESS / PROPERTY-ASCRIPTION  :TENSE PRESENT  :LEX EQUAL :NUMBER MASS :DOMAIN  (ATTRIBUTENAME / SUBJECT :LEX {0} :DETERMINER THE) :RANGE (FIRST / OBJECT :LEX {1}\. :NUMBER MASS)) '.format(
                attribute.upper(), self.first.upper())
            # spl = r'(EQUALNESS / PROPERTY-ASCRIPTION  :TENSE PRESENT  :LEX EQUAL :NUMBER MASS :DOMAIN  (ATTRIBUTENAME / SUBJECT :LEX {0} :DETERMINER THE) :RANGE (FIRST / ABSTRACTION :QUANTITY {1})) '.format(
            #     attribute.upper(), self.first.upper())

        elif self.type == '<<' and self.first == self.second:
            spl = r'(EQUALNESS / PROPERTY-ASCRIPTION  :TENSE PRESENT  :LEX EQUAL :NUMBER MASS :DOMAIN  (ATTRIBUTENAME / SUBJECT :LEX {0} :DETERMINER THE) :RANGE (FIRST / OBJECT :LEX {1}\. :NUMBER MASS)) '.format(
                attribute.upper(), self.first.upper())
            # spl = r'(EQUALNESS / PROPERTY-ASCRIPTION  :TENSE PRESENT  :LEX EQUAL :NUMBER MASS :DOMAIN  (ATTRIBUTENAME / SUBJECT :LEX {0} :DETERMINER THE) :RANGE (FIRST / ABSTRACTION :QUANTITY {1})) '.format(
            #     attribute.upper(), self.first.upper())

        dict = {}
        dict[self.attribute] = {'spl':spl}

        return dict

class Extraction:
    def __init__(self, type, leftSide, extract,rightSide=None):
        self.type = type
        self.leftSide = leftSide
        self.rightSide = rightSide

        self.tree = self.extractTree()

        if extract:
            self.CoNLL = self.extractCoNLL()
            self.DSyntS = self.extractDSyntS()
            self.SPL = self.extractSPL()
        else:
            self.CoNLL = None
            self.DSyntS = None
            self.SPL = self.extractSPL()


    def extractTree(self):
        if self.type == 'and':
            dict = {}
            dict['and'] = [self.leftSide.tree, self.rightSide.tree]

            return dict

        elif self.type == 'or':
            dict = {}
            dict['or'] = [self.leftSide.tree, self.rightSide.tree]

            return dict

        elif self.type == 'imply':
            dict = {}
            dict['imply'] = [self.leftSide.tree, self.rightSide.tree]

            return dict

        elif self.type == 'not':
            dict = {}
            dict['not'] = self.leftSide.tree

            return dict

        elif self.type == 'parentheses':
            dict = {}
            dict['parentheses'] = self.leftSide.tree

            return dict

    def extractCoNLL(self):
        if self.type == 'and':
            dict = {}
            dict['and'] = [self.leftSide.CoNLL, self.rightSide.CoNLL]

            return dict

        elif self.type == 'or':
            dict = {}
            dict['or'] = [self.leftSide.CoNLL, self.rightSide.CoNLL]

            return dict

        elif self.type == 'imply':
            dict = {}
            dict['imply'] = [self.leftSide.CoNLL, self.rightSide.CoNLL]

            return dict

        elif self.type == 'not':
            dict = {}
            dict['not'] = self.leftSide.CoNLL

            return dict

        elif self.type == 'parentheses':
            dict = {}
            dict['parentheses'] = self.leftSide.CoNLL

            return dict

    def extractDSyntS(self):
        if self.type == 'and':
            dict = {}
            dict['and'] = [self.leftSide.DSyntS, self.rightSide.DSyntS]

            return dict

        elif self.type == 'or':
            dict = {}
            dict['or'] = [self.leftSide.DSyntS, self.rightSide.DSyntS]

            return dict

        elif self.type == 'imply':
            dict = {}
            dict['imply'] = [self.leftSide.DSyntS, self.rightSide.DSyntS]

            return dict

        elif self.type == 'not':
            dict = {}
            dict['not'] = self.leftSide.DSyntS

            return dict

        elif self.type == 'parentheses':
            dict = {}
            dict['parentheses'] = self.leftSide.DSyntS

            return dict

    def extractSPL(self):
        if self.type == 'and':
            dict = {}
            dict['and'] = [self.leftSide.SPL, self.rightSide.SPL]

            return dict

        elif self.type == 'or':
            dict = {}
            dict['or'] = [self.leftSide.SPL, self.rightSide.SPL]

            return dict

        elif self.type == 'imply':
            dict = {}
            dict['imply'] = [self.leftSide.SPL, self.rightSide.SPL]

            return dict

        elif self.type == 'not':
            dict = {}
            dict['not'] = self.leftSide.SPL

            return dict

        elif self.type == 'parentheses':
            dict = {}
            dict['parentheses'] = self.leftSide.SPL

            return dict