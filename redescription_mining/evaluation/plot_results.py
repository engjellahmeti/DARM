import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns

# plt.style.use('_mpl-gallery')
plt.rcParams["savefig.format"] = 'pdf'
plt.rcParams.update({'font.size': 36})
sns.set(style="darkgrid")

def plot_results(filename, type):
    with open(r'redescription_mining\evaluation\rule_quality_measures_v2.json', 'r') as a:
        rule_quality_measures = json.load(a)

    count_dc = len(rule_quality_measures[filename][type].keys())
    if count_dc > 0:
        fig, axes = plt.subplots(nrows=6, ncols=count_dc, sharey='row', figsize=(25, 15))
        results = {'AEJ': {'reremi':[], 'splittrees': [], 'new-approach': []}, 'AAJ': {'reremi':[], 'splittrees': [], 'new-approach': []}, 'Confidence': {'reremi':[], 'splittrees': [], 'new-approach': []}
        ,'Inf. Gain': {'reremi':[], 'splittrees': [], 'new-approach': []}
        ,'Lift': {'reremi':[], 'splittrees': [], 'new-approach': []}
        , 'Pearson': {'reremi':[], 'splittrees': [], 'new-approach': []}}

        i = 0
        for dc in rule_quality_measures[filename][type].keys():
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
                results['AEJ']['reremi'].append(_reremi[key]['AEJ'])
                results['AAJ']['reremi'].append(_reremi[key]['AAJ'])
                results['Confidence']['reremi'].append(_reremi[key]['Confidence'])
                results['Inf. Gain']['reremi'].append(_reremi[key]['Information Gain'])
                results['Lift']['reremi'].append(_reremi[key]['Lift'])
                results['Pearson']['reremi'].append(_reremi[key]['Pearson correlation coefficient'])

            for key in _splittrees.keys():
                results['AEJ']['splittrees'].append(_splittrees[key]['AEJ'])
                results['AAJ']['splittrees'].append(_splittrees[key]['AAJ'])
                results['Confidence']['splittrees'].append(_splittrees[key]['Confidence'])
                results['Inf. Gain']['splittrees'].append(_splittrees[key]['Information Gain'])
                results['Lift']['splittrees'].append(_splittrees[key]['Lift'])
                results['Pearson']['splittrees'].append(_splittrees[key]['Pearson correlation coefficient'])

            for key in _new_approach.keys():
                results['AEJ']['new-approach'].append(_new_approach[key]['AEJ'])
                results['AAJ']['new-approach'].append(_new_approach[key]['AAJ'])
                results['Confidence']['new-approach'].append(_new_approach[key]['Confidence'])
                results['Inf. Gain']['new-approach'].append(_new_approach[key]['Information Gain'])
                results['Lift']['new-approach'].append(_new_approach[key]['Lift'])
                results['Pearson']['new-approach'].append(_new_approach[key]['Pearson correlation coefficient'])


            if count_dc == 1:
                for j in range(0, 6):
                    measure_ = list(results.keys())[j]
                    data = [results[measure_]['reremi'], results[measure_]['splittrees'], results[measure_]['new-approach']]   
                    sns.boxplot( data=data,  orient='v' , ax=axes[j], palette="Pastel1")
                    # bp = axes[j].boxplot(data)
                    
                    axes[j].set_xticklabels(['', '', ''])
                    if j ==5:
                        axes[j].set_xticklabels(['ReReMi', 'SplitT', 'New Approach'])
                    
                    # axes[j].set_xticklabels(['ReReMi ({})'.format(len(results[measure_]['reremi'])), 'SplitT ({})'.format(len(results[measure_]['splittrees'])), 'New Approach ({})'.format(len(results[measure_]['new-approach']))])
                    axes[j].set_ylabel(measure_)
                    if j == 0:
                        axes[j].set_title(dc)
            else:
                for j in range(0, 6):
                                        
                    measure_ = list(results.keys())[j]
                    data = [results[measure_]['reremi'], results[measure_]['splittrees'], results[measure_]['new-approach']]   
                    sns.boxplot( data=data,  orient='v' , ax=axes[j][i], palette="Pastel1")
                    # bp = axes[j][i].boxplot(data)
                    
                    axes[j][i].set_xticklabels(['', '', ''])
                    if j ==5:
                        axes[j][i].set_xticklabels(['ReReMi', 'SplitT', 'New Approach'])
                    # axes[j][i].set_xticklabels(['ReReMi ({})'.format(len(results[measure_]['reremi'])), 'SplitT ({})'.format(len(results[measure_]['splittrees'])), 'New Approach ({})'.format(len(results[measure_]['new-approach']))])
                    if j == 0:
                        axes[j][i].set_title(dc)
                    if i == 0:
                        axes[j][i].set_ylabel(measure_)
            i+=1

        plt.savefig(r'redescription_mining\evaluation\plots\{0}-{1}.pdf'.format(filename, type),  dpi=300)
    
