"""
    @Author: Engjëll Ahmeti
    @Date: 11.11.2022
    @LastUpdate: 21.11.2022
"""

import pandas as pd
import h2o
import os
from redescription_mining.new_approach.h2o_decision_tree import H2ODecisionTree

def get_data(activation_path, target_path):
    activation = pd.read_csv(activation_path, index_col=0)
    target = pd.read_csv(target_path, index_col=0)

    activation_dtypes = activation.dtypes
    for key in activation_dtypes.keys():
        _temp = {}
        if str(activation_dtypes[key]) != 'object':
            _temp['dtype'] = activation_dtypes[key]
            _temp['min_val'] = activation[key].min()
            _temp['max_val'] = activation[key].max()
        else:
            _temp['dtype'] = activation_dtypes[key]
            _temp['min_val'] = ''
            _temp['max_val'] = ''
        activation_dtypes[key] = _temp

    target_dtypes = target.dtypes
    for key in target_dtypes.keys():
        _temp = {}
        if str(target_dtypes[key]) != 'object':
            _temp['dtype'] = target_dtypes[key]
            _temp['min_val'] = target[key].min()
            _temp['max_val'] = target[key].max()
        else:
            _temp['dtype'] = target_dtypes[key]
            _temp['min_val'] = ''
            _temp['max_val'] = ''
        target_dtypes[key] = _temp

    return activation, target, activation_dtypes, target_dtypes

def evaluate_h2o_model(df_data, feature_names, target_col, model):
    h2ofr = h2o.H2OFrame(df_data)
    h2ofr.col_names = list(df_data.columns)
    if 'c' in df_data:
        h2ofr['c'] = h2ofr['c'].asfactor() # make categorical
    
    model.train(x=feature_names, y=target_col, training_frame=h2ofr)

    return model

def create_the_tree(model, column_name,negative_or_positive='positive', filename='activation'):
    path = os.path.abspath('redescription_mining/new_approach/experiment')
                            # redescription_mining\new_approach\experiment\results\positive

    if not os.path.exists('{0}\{1}'.format(path, 'results\\' + negative_or_positive)):
        os.makedirs('{0}\{1}'.format(path, 'results\\' + negative_or_positive))

    model_path = '{0}\{1}'.format(path, 'results\\' + negative_or_positive)
    model.model.download_mojo(path=model_path + '\{}.zip'.format(filename))

    os.system('java -cp "{0}\h2o_jar\h2o.jar" hex.genmodel.tools.PrintMojo --tree 0 -i "{1}\{2}.zip" -o "{1}\{2}.gv" -f 20 -d 3'.format(path, model_path, filename))
    
    # os.system('dot -Tpng "{0}\{1}.gv" -o "{0}\{1}-{2}.png"'.format(model_path, filename, column_name))
    os.system('dot -Txdot_json "{0}\{1}.gv" -o "{0}\{1}-{2}.json"'.format(model_path, filename, column_name))

def discover_the_best_tree_one_side(data, y, y_columns, dtypess, negative_or_positive='positive', filename='activation'):
    top_performance = {}
    for column in y_columns:
        data[column + '_as_y'] = y[column]
        
        model = evaluate_h2o_model(data, list(data.columns)[:-1], list(data.columns)[-1], H2ODecisionTree())

        stats = model.all_statistics()
        if stats != 'No model available':
            var_importance = model.varimp()
            model_performance = model.model_performance()

            print(model_performance)

            data.drop([column + '_as_y'], axis=1, inplace=True)
            
            column = column.replace('_as_y', '')
            try:
                top_performance[filename + '-' + column] = {
                        'type_of_tree': filename,
                        'y_column': column,
                        'y_column_dtype': dtypess[column]['dtype'],
                        'y_column_min_val': dtypess[column]['min_val'],
                        'y_column_max_val': dtypess[column]['max_val'],                
                        'variable_importance': var_importance,
                        'model': model,
                        'all_statistics': stats,
                        'model_performance': model_performance
                    }

                create_the_tree(model=model, column_name=column,negative_or_positive=negative_or_positive, filename=filename)
            except:
                print('The found model does not lead to a length of the number of variables!')
    
    return top_performance

def discover_the_best_tree(negative_or_positive, activation_path, target_path):
    activation, target, activation_dtypes, target_dtypes = get_data(activation_path, target_path)

    performance = discover_the_best_tree_one_side(activation, target, target.columns, target_dtypes, negative_or_positive=negative_or_positive, filename='activation')

    performance.update(discover_the_best_tree_one_side(target, activation, activation.columns, activation_dtypes, negative_or_positive=negative_or_positive, filename='target'))

    return performance

