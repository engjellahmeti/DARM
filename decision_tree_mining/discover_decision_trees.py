"""
    @Author: Engjëll Ahmeti
    @Date: 11.11.2022
    @LastUpdate: 16.11.2022
"""

import pandas as pd
import h2o
from decision_tree_mining.h2o_decision_tree import H2ODecisionTree
import os


def get_data(frame_path):
    activation = pd.read_csv(frame_path + r'\activation.csv', index_col=0)
    target = pd.read_csv(frame_path + r'\target.csv', index_col=0)

    return activation, target

def evaluate_h2o_model(df_data, feature_names, target_col, model):
    h2ofr = h2o.H2OFrame(df_data)
    h2ofr.col_names = list(df_data.columns)
    if 'c' in df_data:
        h2ofr['c'] = h2ofr['c'].asfactor() # make categorical
    
    model.train(x=feature_names, y=target_col, training_frame=h2ofr)

    return model

def create_the_tree(model, event_log_name, algorithm, column_name,negative_or_positive='positive', filename='activation'):
    path = os.path.abspath('decision_tree_mining/experiment')
    if not os.path.exists('{0}\{1}-{2}'.format(path, event_log_name, algorithm)):
        os.makedirs('{0}\{1}-{2}'.format(path, event_log_name, algorithm))

    if not os.path.exists('{0}\{1}-{2}\{3}'.format(path, event_log_name, algorithm, negative_or_positive)):
        os.makedirs('{0}\{1}-{2}\{3}'.format(path, event_log_name, algorithm, negative_or_positive))

    model_path = '{0}\{1}-{2}\{3}'.format(path, event_log_name, algorithm, negative_or_positive)
    model.model.download_mojo(path=model_path + '\{}.zip'.format(filename))

    os.system('java -cp "{0}\h2o_jar\h2o.jar" hex.genmodel.tools.PrintMojo --tree 12 -i "{1}\{2}.zip" -o "{1}\{2}.gv" -f 20 -d 3'.format(path, model_path, filename))
    
    os.system('dot -Tpng "{0}\{1}.gv" -o "{0}\{1}-{2}.png"'.format(model_path, filename, column_name))
    os.system('dot -Txdot_json "{0}\{1}.gv" -o "{0}\{1}-{2}.json"'.format(model_path, filename, column_name))

def find_the_best_tree_one_side(data, y, y_columns,  event_log_name, algorithm, negative_or_positive='positive', filename='activation'):
    top_imp = []
    for column in y_columns:
        data[column + '_as_y'] = y[column]
        
        model = evaluate_h2o_model(data, list(data.columns)[:-1], list(data.columns)[-1], H2ODecisionTree())

        stats = model.all_statistics()
        var_importance = model.varimp()
        model_performance = model.model_performance()

        # print(var_importance)
        print(model_performance)

        data.drop([column + '_as_y'], axis=1, inplace=True)

        if len(top_imp) == 0:
            top_imp = [column, var_importance, model]

        else:
            counter = 0
            if len(var_importance) > len(top_imp[1]):
                for variable in top_imp[1]:
                    if top_imp[1][variable] < var_importance[variable]:
                        counter += 1
                
                if counter == len(top_imp[1]):
                    top_imp = [column, var_importance, model]
            else:
                for variable in var_importance:
                    if top_imp[1][variable] < var_importance[variable]:
                        counter += 1
                
                if counter == len(var_importance):
                    top_imp = [column, var_importance, model]

        create_the_tree(model=model, event_log_name=event_log_name, algorithm=algorithm, column_name=column,negative_or_positive=negative_or_positive, filename=filename)
    
    return top_imp


def find_the_best_tree(event_log_name, algorithm, negative_or_positive):
    frame_path = os.path.abspath('feature_vectors\csv_feature_vectors\\' + negative_or_positive)
    activation, target = get_data(frame_path)

    target_column, activation_var_importance, activation_model = find_the_best_tree_one_side(activation, target, target.columns, event_log_name, algorithm, negative_or_positive=negative_or_positive, filename='activation')

    activation_column, target_var_importance, target_model = find_the_best_tree_one_side(target, activation, activation.columns, event_log_name, algorithm, negative_or_positive=negative_or_positive, filename='target')


if __name__ == '__main__':
    filename = 'running-example'
    algorithm = 'reremi'
    negative_or_positive = 'positive'
    h2o.init()
    h2o.no_progress()
    find_the_best_tree(event_log_name=filename, algorithm=algorithm, negative_or_positive=negative_or_positive)
    # create_the_tree(model, filename, algorithm)
    # print(model.model)
    