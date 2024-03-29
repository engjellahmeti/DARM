from event_log_generation.get_declare_constraints import get_declare_constraints
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import math
import matplotlib as mpl
import pandas as pd
import re

# plt.style.use('_mpl-gallery')
plt.rcParams["savefig.format"] = 'pdf'
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
# sns.set(style="darkgrid")
sns.set(style="whitegrid")
with open(r'redescription_mining\evaluation\support.json', 'r') as sup:
    support = json.load(sup)

def helper(filename, type, length, count_dc, rule_quality_measures, keys, name=None):
    fig, axes = plt.subplots(nrows=8, ncols=length, sharey='row', figsize=(10, 10))

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
                    axes[j].set_xticklabels(['ReReMi ({})'.format(len(results[measure_]['reremi'])), 'SplitT ({})'.format(len(results[measure_]['splittrees'])), 'RF-SplitT ({})'.format(len(results[measure_]['new-approach']))])
                
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
                    axes[j][i].set_xticklabels(['ReReMi ({})'.format(len(results[measure_]['reremi'])), 'SplitT ({})'.format(len(results[measure_]['splittrees'])), 'RF-SplitT ({})'.format(len(results[measure_]['new-approach']))])
                if j == 0:
                    axes[j][i].set_title(dc + ' - {} entities'.format(get_no_entities))
                if i == 0:
                    axes[j][i].set_ylabel(measure_)
        i+=1

    bbox_inches="tight"

    if name:
        plt.savefig(r'redescription_mining\evaluation\plots\{0}-{1}-{2}.pdf'.format(filename, type, name),  bbox_inches='tight')
    else:
        plt.savefig(r'redescription_mining\evaluation\plots\{0}-{1}.pdf'.format(filename, type),   bbox_inches='tight')

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

def plot_results_improved(filenames, type):
    with open(r'redescription_mining\evaluation\rule_quality_measures_v2.json', 'r') as a:
        rule_quality_measures = json.load(a)
    measurements = ['Jaccard Index', 'p-value', 'AEJ', 'AAJ', 'Confidence', 'Inf. Gain', 'Lift', 'Pearson']

    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    fig = plt.figure()
    gs = fig.add_gridspec(len(measurements), hspace=0)
    axs = gs.subplots(sharex=True)
    for j, measure_ in enumerate(measurements):
        df = pd.DataFrame(columns=['Constraint', 'Algorithm', 'Jaccard Index', 'p-value', 'AEJ', 'AAJ', 'Confidence', 'Inf. Gain', 'Lift', 'Pearson'])
        i = 0
        filename = filenames[0]
        print_name = filenames[1]
        constraints = get_declare_constraints(declare_file_path='event_log_generation\declare constraint files\{}.decl'.format(print_name))

        for dc in constraints:
            dc = dc.str_representation() 

            dc1 = dc
            # dc = dc + '- {}'.format(get_no_entities)
            dc1 = re.sub(r'precedence', 'prec', dc1)
            dc1 = re.sub(r'response', 'resp', dc1)
            dc1 = re.sub(r'respondedexistence', 'respExis', dc1)
           
            _reremi = {}
            try:
                if 'reremi' in rule_quality_measures[filename][type][dc].keys():
                    _reremi =  rule_quality_measures[filename][type][dc]['reremi']
                    for key in _reremi.keys():
                        df.loc[i] = [dc1, 'ReReMi', _reremi[key]['Jaccard Index'], _reremi[key]['p-value'], _reremi[key]['AEJ'], _reremi[key]['AAJ'], _reremi[key]['Confidence'], _reremi[key]['Information Gain'], _reremi[key]['Lift'], _reremi[key]['Pearson correlation coefficient']]
                        i+=1
            except:
                pass

            _new_approach = {}
            try:
                if 'new-approach' in rule_quality_measures[filename][type][dc].keys():
                    _new_approach =  rule_quality_measures[filename][type][dc]['new-approach']
                    for key in _new_approach.keys():
                        df.loc[i] = [dc1, 'RF-SplitT',  _new_approach[key]['Jaccard Index'], _new_approach[key]['p-value'], _new_approach[key]['AEJ'], _new_approach[key]['AAJ'], _new_approach[key]['Confidence'], _new_approach[key]['Information Gain'], _new_approach[key]['Lift'], _new_approach[key]['Pearson correlation coefficient']]
                        i+=1
            except:
                pass

            _splittrees = {}
            try:
                if 'splittrees' in rule_quality_measures[filename][type][dc].keys():
                    _splittrees =  rule_quality_measures[filename][type][dc]['splittrees']
                    for key in _splittrees.keys():
                        df.loc[i] = [dc1, 'SplitT',  _splittrees[key]['Jaccard Index'], _splittrees[key]['p-value'], _splittrees[key]['AEJ'], _splittrees[key]['AAJ'], _splittrees[key]['Confidence'], _splittrees[key]['Information Gain'], _splittrees[key]['Lift'], _splittrees[key]['Pearson correlation coefficient']]
                        i+=1
            except:
                pass


            try:
                temp = '{0}-reremi'.format(filename)
                get_no_entities = support[temp][type][list(_reremi.keys())[0]]['|V|']
            except:
                try:
                    if get_no_entities == -1:
                        get_no_entities = support['{0}-splittrees'.format(filename)][type][list(_splittrees.keys())[0]]['|V|']
                except:
                    try:
                        if get_no_entities == -1:
                            get_no_entities = support['{0}-new-approach'.format(filename)][type][list(_new_approach.keys())[0]]['|V|']
                    except:
                        get_no_entities = -1

            if df[(df['Constraint'] == dc) & (df['Algorithm'] == 'ReReMi')].empty:
                df.loc[i] = [dc1, 'ReReMi', None,None,None,None,None,None,None,None]
                i+=1
            if df[(df['Constraint'] == dc) & (df['Algorithm'] == 'SplitT')].empty:
                df.loc[i] = [dc1, 'SplitT', None,None,None,None,None,None,None,None]
                i+=1
            if df[(df['Constraint'] == dc) & (df['Algorithm'] == 'RF-SplitT')].empty:
                df.loc[i] = [dc1, 'RF-SplitT', None,None,None,None,None,None,None,None]
                i+=1

         
        
        # https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
        # flare
        # sns.boxplot(data=df, x="Algorithm", y=measure_, hue='Constraint', orient='v', palette="flare", ax=axs[j])

        # light sequential gradient 
        # sns.boxplot(data=df, x="Algorithm", y=measure_, hue='Constraint', orient='v', palette="light:#5A9", ax=axs[j])

        # pastel
        # sns.boxplot(data=df, x="Algorithm", y=measure_, hue='Constraint', orient='v', palette="pastel", ax=axs[j])

        # Spectral
        sns.boxplot(data=df, x="Algorithm", y=measure_, hue='Constraint', orient='v', palette="Spectral", ax=axs[j])

    for i, ax in enumerate(axs):
        if i != len(axs) - 1:
            ax.get_legend().remove()
        else:
            # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=True, ncol=5)
        ax.label_outer()
        ax.grid(False)

    if 'unning' in print_name:
        print_name = 'Order-To-Cash'
    elif 'Credit' in print_name:
        print_name = 'Credit Application'
    axs[0].set_title(print_name)
    plt.xlabel('')
    import matplotlib
    plot_backend = matplotlib.get_backend()
    mng = plt.get_current_fig_manager()
    if plot_backend == 'TkAgg':
        mng.resize(*mng.window.maxsize())
    elif plot_backend == 'wxAgg':
        mng.frame.Maximize(True)
    elif plot_backend == 'Qt4Agg':
        mng.window.showMaximized()

    plt.show()
    print()

def plot_results_per_declare_constraint(filenames, type):
    with open(r'redescription_mining\evaluation\rule_quality_measures_v2.json', 'r') as a:
        rule_quality_measures = json.load(a)
    measurements = ['Jaccard Index', 'p-value', 'AEJ', 'AAJ', 'Confidence', 'Inf. Gain', 'Lift', 'Pearson']

    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    fig = plt.figure()
    gs = fig.add_gridspec(len(measurements), hspace=0)
    axs = gs.subplots(sharex=True)
    filename = filenames[0]
    print_name = filenames[1]
    dc = filenames[2]
    dc = dc.str_representation() 
    for j, measure_ in enumerate(measurements):
        df = pd.DataFrame(columns=['Constraint', 'Algorithm', 'Algorithms', 'Jaccard Index', 'p-value', 'AEJ', 'AAJ', 'Confidence', 'Inf. Gain', 'Lift', 'Pearson'])
        i = 0


        dc1 = dc
        dc1 = re.sub(r'precedence', 'prec', dc1)
        dc1 = re.sub(r'response', 'resp', dc1)
        dc1 = re.sub(r'respondedexistence', 'respExis', dc1)
        
        _reremi = {}
        try:
            if 'reremi' in rule_quality_measures[filename][type][dc].keys():
                _reremi =  rule_quality_measures[filename][type][dc]['reremi']
                for key in _reremi.keys():
                    df.loc[i] = [dc1, 'ReReMi', 'ReReMi ({})'.format(len(_reremi.keys())), _reremi[key]['Jaccard Index'], _reremi[key]['p-value'], _reremi[key]['AEJ'], _reremi[key]['AAJ'], _reremi[key]['Confidence'], _reremi[key]['Information Gain'], _reremi[key]['Lift'], _reremi[key]['Pearson correlation coefficient']]
                    i+=1
            else:
                df.loc[i] = [dc1, 'ReReMi', 'ReReMi (0)', None,None,None,None,None,None,None,None]
                i+=1
        except:
            pass

        _new_approach = {}
        try:
            if 'new-approach' in rule_quality_measures[filename][type][dc].keys():
                _new_approach =  rule_quality_measures[filename][type][dc]['new-approach']
                for key in _new_approach.keys():
                    df.loc[i] = [dc1, 'RF-SplitT', 'RF-SplitT ({})'.format(len(_new_approach.keys())),  _new_approach[key]['Jaccard Index'], _new_approach[key]['p-value'], _new_approach[key]['AEJ'], _new_approach[key]['AAJ'], _new_approach[key]['Confidence'], _new_approach[key]['Information Gain'], _new_approach[key]['Lift'], _new_approach[key]['Pearson correlation coefficient']]
                    i+=1
            else:
                df.loc[i] = [dc1, 'RF-SplitT', 'RF-SplitT (0)', None,None,None,None,None,None,None,None]
                i+=1
        except:
            pass

        _splittrees = {}
        try:
            if 'splittrees' in rule_quality_measures[filename][type][dc].keys():
                _splittrees =  rule_quality_measures[filename][type][dc]['splittrees']
                for key in _splittrees.keys():
                    df.loc[i] = [dc1, 'SplitT', 'SplitT ({})'.format(len(_splittrees.keys())),  _splittrees[key]['Jaccard Index'], _splittrees[key]['p-value'], _splittrees[key]['AEJ'], _splittrees[key]['AAJ'], _splittrees[key]['Confidence'], _splittrees[key]['Information Gain'], _splittrees[key]['Lift'], _splittrees[key]['Pearson correlation coefficient']]
                    i+=1
            else:
                df.loc[i] = [dc1, 'SplitT', 'SplitT (0)', None,None,None,None,None,None,None,None]
                i+=1
        except:
            pass

        try:
            temp = '{0}-reremi'.format(filename)
            get_no_entities = support[temp][type][list(_reremi.keys())[0]]['|V|']
        except:
            try:
                if get_no_entities == -1:
                    get_no_entities = support['{0}-splittrees'.format(filename)][type][list(_splittrees.keys())[0]]['|V|']
            except:
                try:
                    if get_no_entities == -1:
                        get_no_entities = support['{0}-new-approach'.format(filename)][type][list(_new_approach.keys())[0]]['|V|']
                except:
                    get_no_entities = -1

        if df[(df['Constraint'] == dc) & (df['Algorithm'] == 'ReReMi')].empty:
            df.loc[i] = [dc1, 'ReReMi', 'ReReMi ({})'.format(len(_reremi.keys())), None,None,None,None,None,None,None,None]
            i+=1
        if df[(df['Constraint'] == dc) & (df['Algorithm'] == 'SplitT')].empty:
            df.loc[i] = [dc1, 'SplitT', 'SplitT ({})'.format(len(_splittrees.keys())), None,None,None,None,None,None,None,None]
            i+=1
        if df[(df['Constraint'] == dc) & (df['Algorithm'] == 'RF-SplitT')].empty:
            df.loc[i] = [dc1, 'RF-SplitT', 'RF-SplitT ({})'.format(len(_new_approach.keys())), None,None,None,None,None,None,None,None]
            i+=1

        sns.boxplot(data=df, x="Algorithms", y=measure_, ax=axs[j])

    if 'unning' in print_name:
        print_name = 'Order-To-Cash'
    elif 'Credit' in print_name:
        print_name = 'Credit Application'
    axs[0].set_title(print_name + ' - ' + dc.__str__())
    plt.xlabel('')
    import matplotlib
    plot_backend = matplotlib.get_backend()
    mng = plt.get_current_fig_manager()
    if plot_backend == 'TkAgg':
        mng.resize(*mng.window.maxsize())
    elif plot_backend == 'wxAgg':
        mng.frame.Maximize(True)
    elif plot_backend == 'Qt4Agg':
        mng.window.showMaximized()

    print(filename, type, dc)
    plt.show()
    # plt.savefig(r'redescription_mining\evaluation\plots\{0}-{1}-{2}.pdf'.format(filename, type, dc),  bbox_inches='tight')

    print()

def check_key(dict, dc, algorithm):
    try:
        return dict[dc][algorithm]
    except:
        return {}

def check_key_all(dict, print_name, algorithm):
    constraints = get_declare_constraints(declare_file_path='event_log_generation\declare constraint files\{}.decl'.format(print_name))
    i = 0
    for dc in constraints:
        dc = dc.str_representation()
        try:
            i += len(dict[dc][algorithm].keys())
        except:
            i += 0
    return i

def plot_results_per_declare_constraint_negative_positive(filenames):
    with open(r'redescription_mining\evaluation\rule_quality_measures_v2.json', 'r') as a:
        rule_quality_measures = json.load(a)
    measurements = ['Jaccard Index', 'p-value', 'AEJ', 'AAJ', 'Inf. Gain', 'Lift', 'Pearson']

    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    fig = plt.figure()
    gs = fig.add_gridspec(len(measurements), hspace=0)
    axs = gs.subplots(sharex=True)
    filename = filenames[0]
    print_name = filenames[1]
    dc = filenames[2]
    dc = dc.str_representation() 
    reremi_neg = check_key(rule_quality_measures[filename]['negative'], dc, 'reremi')
    reremi_pos = check_key(rule_quality_measures[filename]['positive'], dc, 'reremi')
    rfsplitt_neg = check_key(rule_quality_measures[filename]['negative'], dc, 'new-approach')
    rfsplitt_pos = check_key(rule_quality_measures[filename]['positive'], dc, 'new-approach')
    splittrees_neg = check_key(rule_quality_measures[filename]['negative'], dc, 'splittrees')
    splittrees_pos = check_key(rule_quality_measures[filename]['positive'], dc, 'splittrees')
    for j, measure_ in enumerate(measurements):
        i = 0
        df = pd.DataFrame(columns=['Constraint', 'Algorithm', 'Algorithms', 'Type', 'Jaccard Index', 'p-value', 'AEJ', 'AAJ',  'Inf. Gain', 'Lift', 'Pearson'])
        for type in ['positive', 'negative']:


            dc1 = dc
            dc1 = re.sub(r'precedence', 'prec', dc1)
            dc1 = re.sub(r'response', 'resp', dc1)
            dc1 = re.sub(r'respondedexistence', 'respExis', dc1)
            
            _reremi = {}
            try:
                if 'reremi' in rule_quality_measures[filename][type][dc].keys():
                    _reremi =  rule_quality_measures[filename][type][dc]['reremi']
                    for key in _reremi.keys():
                        df.loc[i] = [dc1, 'ReReMi', 'ReReMi ({0}, {1})'.format(len(reremi_pos.keys()), len(reremi_neg.keys())), 'Positive' if type == 'positive' else 'Deviant', _reremi[key]['Jaccard Index'], _reremi[key]['p-value'], _reremi[key]['AEJ'], _reremi[key]['AAJ'],  _reremi[key]['Information Gain'], _reremi[key]['Lift'], _reremi[key]['Pearson correlation coefficient']]
                        i+=1
                else:
                    df.loc[i] = [dc1, 'ReReMi', 'ReReMi ({0}, {1})'.format(len(reremi_pos.keys()), len(reremi_neg.keys())), 'Positive' if type == 'positive' else 'Deviant', None,None,None,None,None,None,None]
                    i+=1
            except:
                pass

            _new_approach = {}
            try:
                if 'new-approach' in rule_quality_measures[filename][type][dc].keys():
                    _new_approach =  rule_quality_measures[filename][type][dc]['new-approach']
                    for key in _new_approach.keys():
                        df.loc[i] = [dc1, 'RF-SplitT', 'RF-SplitT ({0}, {1})'.format(len(rfsplitt_pos.keys()), len(rfsplitt_neg.keys())), 'Positive' if type == 'positive' else 'Deviant',  _new_approach[key]['Jaccard Index'], _new_approach[key]['p-value'], _new_approach[key]['AEJ'], _new_approach[key]['AAJ'], _new_approach[key]['Information Gain'], _new_approach[key]['Lift'], _new_approach[key]['Pearson correlation coefficient']]
                        i+=1
                else:
                    df.loc[i] = [dc1, 'RF-SplitT', 'RF-SplitT ({0}, {1})'.format(len(rfsplitt_pos.keys()), len(rfsplitt_neg.keys())), 'Positive' if type == 'positive' else 'Deviant', None,None,None,None,None,None,None]
                    i+=1
            except:
                pass

            _splittrees = {}
            try:
                if 'splittrees' in rule_quality_measures[filename][type][dc].keys():
                    _splittrees =  rule_quality_measures[filename][type][dc]['splittrees']
                    for key in _splittrees.keys():
                        df.loc[i] = [dc1, 'SplitT', 'SplitT ({0}, {1})'.format(len(splittrees_pos.keys()), len(splittrees_neg.keys())), 'Positive' if type == 'positive' else 'Deviant',  _splittrees[key]['Jaccard Index'], _splittrees[key]['p-value'], _splittrees[key]['AEJ'], _splittrees[key]['AAJ'], _splittrees[key]['Information Gain'], _splittrees[key]['Lift'], _splittrees[key]['Pearson correlation coefficient']]
                        i+=1
                else:
                    df.loc[i] = [dc1, 'SplitT', 'SplitT ({0}, {1})'.format(len(splittrees_pos.keys()), len(splittrees_neg.keys())), 'Positive' if type == 'positive' else 'Deviant', None,None,None,None,None,None,None]
                    i+=1
            except:
                pass

            try:
                temp = '{0}-reremi'.format(filename)
                get_no_entities = support[temp][type][list(_reremi.keys())[0]]['|V|']
            except:
                try:
                    if get_no_entities == -1:
                        get_no_entities = support['{0}-splittrees'.format(filename)][type][list(_splittrees.keys())[0]]['|V|']
                except:
                    try:
                        if get_no_entities == -1:
                            get_no_entities = support['{0}-new-approach'.format(filename)][type][list(_new_approach.keys())[0]]['|V|']
                    except:
                        get_no_entities = -1

            if df[(df['Constraint'] == dc) & (df['Algorithm'] == 'ReReMi')].empty:
                df.loc[i] = [dc1, 'ReReMi', 'ReReMi ({0}, {1})'.format(len(reremi_pos.keys()), len(reremi_neg.keys())), 'Positive' if type == 'positive' else 'Deviant', None,None,None,None,None,None,None]
                i+=1
            if df[(df['Constraint'] == dc) & (df['Algorithm'] == 'SplitT')].empty:
                df.loc[i] = [dc1, 'SplitT', 'SplitT ({0}, {1})'.format(len(splittrees_pos.keys()), len(splittrees_neg.keys())), 'Positive' if type == 'positive' else 'Deviant', None,None,None,None,None,None,None]
                i+=1
            if df[(df['Constraint'] == dc) & (df['Algorithm'] == 'RF-SplitT')].empty:
                df.loc[i] = [dc1, 'RF-SplitT', 'RF-SplitT ({0}, {1})'.format(len(rfsplitt_pos.keys()), len(rfsplitt_neg.keys())), 'Positive' if type == 'positive' else 'Deviant', None,None,None,None,None,None,None]
                i+=1

        sns.boxplot(data=df, x="Algorithms", y=measure_, hue="Type", ax=axs[j])

    for i, ax in enumerate(axs):
        if i != len(axs) - 1:
            ax.get_legend().remove()
        else:
            # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=True, ncol=5)
        ax.label_outer()
        ax.grid(False)

    if 'unning' in print_name:
        print_name = 'Order-To-Cash'
    elif 'Credit' in print_name:
        print_name = 'Credit Application'
    axs[0].set_title(print_name + ' - ' + dc.__str__())
    plt.xlabel('')
    import matplotlib
    plot_backend = matplotlib.get_backend()
    mng = plt.get_current_fig_manager()
    if plot_backend == 'TkAgg':
        mng.resize(*mng.window.maxsize())
    elif plot_backend == 'wxAgg':
        mng.frame.Maximize(True)
    elif plot_backend == 'Qt4Agg':
        mng.window.showMaximized()

    print(filename, type, dc)
    plt.show()
    # plt.savefig(r'redescription_mining\evaluation\plots\{0}-{1}-{2}.pdf'.format(filename, type, dc),  bbox_inches='tight')
    print()
  
def get_values(filenames):
    with open(r'redescription_mining\evaluation\rule_quality_measures_v2.json', 'r') as a:
        rule_quality_measures = json.load(a)
    measurements = ['Jaccard Index', 'p-value', 'AEJ', 'AAJ', 'Confidence', 'Inf. Gain', 'Lift', 'Pearson']

    filename = filenames[0]
    print_name = filenames[1]
    reremi_neg = check_key_all(rule_quality_measures[filename]['negative'], print_name, 'reremi')
    reremi_pos = check_key_all(rule_quality_measures[filename]['positive'], print_name, 'reremi')
    rfsplitt_neg = check_key_all(rule_quality_measures[filename]['negative'], print_name, 'new-approach')
    rfsplitt_pos = check_key_all(rule_quality_measures[filename]['positive'], print_name, 'new-approach')
    splittrees_neg = check_key_all(rule_quality_measures[filename]['negative'], print_name, 'splittrees')
    splittrees_pos = check_key_all(rule_quality_measures[filename]['positive'], print_name, 'splittrees')

    print(filename)
    print('    positive')
    print('        ReReMi    RF-SplitT    SplitT')
    print('         {0}        {1}          {2}'.format(reremi_pos, rfsplitt_pos, splittrees_pos))
    
    print(filename)
    print('    negative')
    print('        ReReMi    RF-SplitT    SplitT')
    print('         {0}        {1}          {2}'.format(reremi_neg, rfsplitt_neg, splittrees_neg))
    

def plot_entities(filenames):
    with open(r'redescription_mining\evaluation\support.json', 'r') as a:
        rule_quality_measures = json.load(a)

    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
   
    filename = filenames[0]
    print_name = filenames[1]
    constraints = get_declare_constraints(declare_file_path='event_log_generation\declare constraint files\{}.decl'.format(print_name))
    temp_dict = {'Declare Constraint': [], 'Type': [], 'No. of entities': []}
    for type in  ['negative', 'positive']:
        for dc in constraints:
            dc = dc.str_representation() 
            temp_dict['Declare Constraint'].append(dc)
           
            df = pd.read_csv(r'feature_vectors\csv_feature_vectors\{0}\activation-{1}-{2}.csv'.format(type, filename, dc), index_col=0)
            
            temp_dict['No. of entities'].append(df.shape[0])
            temp_dict['Type'].append('Positive' if type =='positive' else 'Deviant')

            

    data = pd.DataFrame.from_dict(temp_dict)
    sns.barplot(data=data, x='Declare Constraint', y='No. of entities', hue='Type',palette="Spectral")
    plt.xticks(rotation=45)
    if 'unning' in print_name:
        print_name = 'Order-To-Cash'
    elif 'Credit' in print_name:
        print_name = 'Credit Application'
    plt.title(print_name)
    plt.show()
    print()