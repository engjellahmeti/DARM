import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import math

# plt.style.use('_mpl-gallery')
plt.rcParams["savefig.format"] = 'pdf'
plt.rcParams.update({'font.size': 36})
# sns.set(style="darkgrid")
sns.set(style="whitegrid")
with open(r'redescription_mining\evaluation\support.json', 'r') as sup:
    support = json.load(sup)

def helper(filename, type, length, count_dc, rule_quality_measures, keys, name=None):
    fig, axes = plt.subplots(nrows=8, ncols=length, sharey='row', figsize=(25, 15))

    i = 0
    for dc in keys:
        results = {
            'Jaccard Index': {'reremi':[], 'splittrees': [], 'new-approach': []},
            'p-value': {'reremi':[], 'splittrees': [], 'new-approach': []},
            'AEJ': {'reremi':[], 'splittrees': [], 'new-approach': []}, 'AAJ': {'reremi':[], 'splittrees': [], 'new-approach': []}, 'Confidence': {'reremi':[], 'splittrees': [], 'new-approach': []}
        ,'Inf. Gain': {'reremi':[], 'splittrees': [], 'new-approach': []}
        ,'Lift': {'reremi':[], 'splittrees': [], 'new-approach': []}
        , 'Pearson': {'reremi':[], 'splittrees': [], 'new-approach': []}}
        _reremi = {}
        if 'reremi' in rule_quality_measures[filename][type][dc].keys():
            _reremi =  rule_quality_measures[filename][type][dc]['reremi']

        _splittrees = {}
        if 'splittrees' in rule_quality_measures[filename][type][dc].keys():
            _splittrees =  rule_quality_measures[filename][type][dc]['splittrees']

        _new_approach = {}
        if 'new-approach' in rule_quality_measures[filename][type][dc].keys():
            _new_approach =  rule_quality_measures[filename][type][dc]['new-approach']


        for key in _reremi.keys():
            results['Jaccard Index']['reremi'].append(_reremi[key]['Jaccard Index'])
            results['p-value']['reremi'].append(_reremi[key]['p-value'])
            results['AEJ']['reremi'].append(_reremi[key]['AEJ'])
            results['AAJ']['reremi'].append(_reremi[key]['AAJ'])
            results['Confidence']['reremi'].append(_reremi[key]['Confidence'])
            results['Inf. Gain']['reremi'].append(_reremi[key]['Information Gain'])
            results['Lift']['reremi'].append(_reremi[key]['Lift'])
            results['Pearson']['reremi'].append(_reremi[key]['Pearson correlation coefficient'])

        try:
            temp = '{0}-reremi'.format(filename)
            get_no_entities = support[temp][type][key]['|V|']
        except:
            get_no_entities = -1
            
        for key in _splittrees.keys():
            results['Jaccard Index']['splittrees'].append(_splittrees[key]['Jaccard Index'])
            results['p-value']['splittrees'].append(_splittrees[key]['p-value'])
            results['AEJ']['splittrees'].append(_splittrees[key]['AEJ'])
            results['AAJ']['splittrees'].append(_splittrees[key]['AAJ'])
            results['Confidence']['splittrees'].append(_splittrees[key]['Confidence'])
            results['Inf. Gain']['splittrees'].append(_splittrees[key]['Information Gain'])
            results['Lift']['splittrees'].append(_splittrees[key]['Lift'])
            results['Pearson']['splittrees'].append(_splittrees[key]['Pearson correlation coefficient'])

        try:
            if get_no_entities == -1:
                get_no_entities = support['{0}-splittrees'.format(filename)][type][key]['|V|']
        except:
            get_no_entities = -1

        for key in _new_approach.keys():
            results['Jaccard Index']['new-approach'].append(_new_approach[key]['Jaccard Index'])
            results['p-value']['new-approach'].append(_new_approach[key]['p-value'])
            results['AEJ']['new-approach'].append(_new_approach[key]['AEJ'])
            results['AAJ']['new-approach'].append(_new_approach[key]['AAJ'])
            results['Confidence']['new-approach'].append(_new_approach[key]['Confidence'])
            results['Inf. Gain']['new-approach'].append(_new_approach[key]['Information Gain'])
            results['Lift']['new-approach'].append(_new_approach[key]['Lift'])
            results['Pearson']['new-approach'].append(_new_approach[key]['Pearson correlation coefficient'])

        try:
            if get_no_entities == -1:
                get_no_entities = support['{0}-new-approach'.format(filename)][type][key]['|V|']
        except:
            get_no_entities = -1

        if count_dc == 1:
            for j in range(0, 8):
                measure_ = list(results.keys())[j]
                data = [results[measure_]['reremi'], results[measure_]['splittrees'], results[measure_]['new-approach']]   
                sns.boxplot( data=data,  orient='v' , ax=axes[j], palette="Pastel1")
                sns.scatterplot(x=0, y=results[measure_]['reremi'], ax=axes[j])#, palette="Pastel1")
                sns.scatterplot(x=1, y=results[measure_]['splittrees'], ax=axes[j])#, palette="Pastel1")
                sns.scatterplot(x=2, y=results[measure_]['new-approach'], ax=axes[j])#, palette="Pastel1")
                # bp = axes[j].boxplot(data)
                
                axes[j].set_xticklabels(['', '', ''])
                if j == 7:
                    # axes[j].set_xticklabels(['ReReMi', 'SplitT', 'New Approach'])
                    axes[j].set_xticklabels(['ReReMi ({})'.format(len(results[measure_]['reremi'])), 'SplitT ({})'.format(len(results[measure_]['splittrees'])), 'New Approach ({})'.format(len(results[measure_]['new-approach']))])
                
                axes[j].set_ylabel(measure_)
                if j == 0:
                    axes[j].set_title(dc + '- {} entities'.format(get_no_entities))
        else:
            for j in range(0, 8):
                                    
                measure_ = list(results.keys())[j]
                data = [results[measure_]['reremi'], results[measure_]['splittrees'], results[measure_]['new-approach']]   
                sns.boxplot( data=data,  orient='v' , ax=axes[j][i], palette="Pastel1")
                sns.scatterplot(x=0, y=results[measure_]['reremi'], ax=axes[j][i])#, palette="Pastel1")
                sns.scatterplot(x=1, y=results[measure_]['splittrees'], ax=axes[j][i])#, palette="Pastel1")
                sns.scatterplot(x=2, y=results[measure_]['new-approach'], ax=axes[j][i])#, palette="Pastel1")
                # bp = axes[j][i].boxplot(data)
                
                axes[j][i].set_xticklabels(['', '', ''])
                if j == 7:
                    # axes[j][i].set_xticklabels(['ReReMi', 'SplitT', 'New Approach'])
                    axes[j][i].set_xticklabels(['ReReMi ({})'.format(len(results[measure_]['reremi'])), 'SplitT ({})'.format(len(results[measure_]['splittrees'])), 'New Approach ({})'.format(len(results[measure_]['new-approach']))])
                if j == 0:
                    axes[j][i].set_title(dc + ' - {} entities'.format(get_no_entities))
                if i == 0:
                    axes[j][i].set_ylabel(measure_)
        i+=1

    if name:
        plt.savefig(r'redescription_mining\evaluation\plots\{0}-{1}-{2}.pdf'.format(filename, type, name),  dpi=300)
    else:
        plt.savefig(r'redescription_mining\evaluation\plots\{0}-{1}.pdf'.format(filename, type),  dpi=300)

def plot_results(filename, type):
    with open(r'redescription_mining\evaluation\rule_quality_measures_v2.json', 'r') as a:
        rule_quality_measures = json.load(a)
    
    count_dc = len(rule_quality_measures[filename][type].keys())
    if count_dc > 0:
        if count_dc > 3:
        
            for i in range(0, math.ceil(count_dc/3)):
                upper = (i+1)*3 if (i+1)*3 < count_dc else  count_dc
                length = 3 if (i+1)*3 < count_dc else count_dc%3
                helper(filename=filename, type=type, length=length, count_dc=count_dc, rule_quality_measures=rule_quality_measures, keys=list(rule_quality_measures[filename][type].keys())[i*3:upper], name=i+1)
        else:
            helper(filename=filename, type=type, length=count_dc, count_dc=count_dc, rule_quality_measures=rule_quality_measures, keys=rule_quality_measures[filename][type].keys())


    
