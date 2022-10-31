# @project Deviance Analysis by Means of Redescription Mining - Master Thesis 
# @author Engjëll Ahmeti
# @date 1/30/2021
import json

class DSyntS:
    def __init__(self, item):
        self.id = item['ID']
        self.text = item['TEXT']
        self.lemma = item['LEMMA']
        self.pos = item['POS']
        self.ppos = item['PPOS']
        self.feat = item['FEAT']
        self.head = item['HEAD']
        self.deprel = item['DEPREL']

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        word_dict = {}
        word_dict['id'] = self.id
        word_dict['text'] = self.text
        word_dict['lemma'] = self.lemma
        word_dict['pos'] = self.pos
        word_dict['ppos'] = self.ppos
        word_dict['feat'] = self.feat
        word_dict['head'] = self.head
        word_dict['deprel'] = self.deprel

        return word_dict
