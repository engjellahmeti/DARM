"""
    @Author: Engjëll Ahmeti
    @Date: 11.11.2022
    @LastUpdate: 16.11.2022
"""

from h2o.estimators.random_forest import H2ORandomForestEstimator

class H2ODecisionTree:                                                              
    def __init__(self):
        self.model = None

    def train(self, x, y, training_frame):
        self.model = H2ORandomForestEstimator(mtries=len(x), max_depth=len(x), nfolds=10)
        self.model.train(x=x, y=y, training_frame=training_frame)

    def predict(self, frame):
        return self.model.predict(frame)
    
    def varimp(self):        
        return dict([(v[0], v[3]) for v in self.model.varimp()])

    def all_statistics(self):
        return self.model._str_items()

    def model_performance(self):
        return self.model.model_performance()