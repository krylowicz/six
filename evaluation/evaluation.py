from sdv.evaluation import evaluate 
import numpy as np
from sdv.metrics.tabular import LogisticDetection,NumericalMLP,CSTest,KSTest,MulticlassMLPClassifier
from sdv.metrics.timeseries import LSTMDetection, TSFCDetection,TSFClassifierEfficacy
from prettytable import PrettyTable

class Evaluation:
    
    def __init__(self, synthetic, real):
        self.synthetic, self.real = synthetic, real
        
        
    def metric(self,*args, a):
        
        """takes one or more metric names
           a: boolean variable
           returns either a table with the scores of the listed metrics (a=False) or the
           calculated overall score (a=True)
        """
        score=evaluate(self.synthetic, self.real, metrics=list(args), aggregate=a)
        print(score)  
        
    def show_options(self):
        options=["LogisticDetection","SVCDetection",
                    "BinaryDecisionTreeClassifier",
                    "BinaryAdaBoostClassifier",
                    "BinaryLogisticRegression",
                    "BinaryMLPClassifier",
                    "MulticlassDecisionTreeClassifier",
                    "MulticlassMLPClassifier",
                    "LinearRegression",
                    "MLPRegressor",
                    "GMLogLikelihood",
                    "CSTest",
                    "KSTest",
                    "KSTestExtended",
                    "CategoricalCAP",
                    "CategoricalZeroCAP",
                    "CategoricalGeneralizedCAP",
                    "CategoricalNB",
                    "CategoricalKNN",
                    "CategoricalRF",
                    "CategoricalSVM",
                    "CategoricalEnsemble",
                    "NumericalLR",
                    "NumericalMLP",
                    "NumericalSVR",
                    "NumericalRadiusNearestNeighbor",
                    "ContinuousKLDivergence",
                    "DiscreteKLDivergence"]
        for i in options:
                print(i)
                
    
    def table(self,target_column, keys):        
        
        """
          target_colum = the column an attacker will try to predict
          keys = list of columns an attacker will use for predicting the target column (if the target column is categorical then the keys need to be categorical)     
          r: gives overall score on the given tests, this constitutes the resemblance score.
          u: trains MulticlassMLPClassifier on the real data and later uses it to evaluate the syntethic data (TRTS)
             This gives the utility score.
          p: Gives the privacy score with the NumericalMLP method.
          
        """ 
        
        r1=CSTest.compute(self.real,self.synthetic)
        r2=KSTest.compute(self.real,self.synthetic)
        r3=LogisticDetection.compute(self.real,self.synthetic)
        u=MulticlassMLPClassifier.compute(self.real,self.synthetic,target=target_column)
        p=NumericalMLP.compute(self.real, self.synthetic,sensitive_fields=[target_column],key_fields=keys)
        
        
        evaluation_table=PrettyTable()
        evaluation_table.title="SCORES"          
        evaluation_table.field_names=["Resemblance", "Utility (TRTS)","Privacy"]
        evaluation_table.add_row([round((r1+r2+r3)/3,2),round(u,2),round(p,2)])
        
        
        
        evaluation_b=PrettyTable()
        evaluation_b.title="SCORES BREAKDOWN"          
        evaluation_b.field_names=["Resemblance", "R_scores","Utility","Privacy"]
        evaluation_b.add_row(["Chi-S-Test\n K-S-Test\n LogisticDetection","{}\n{}\n{}".format(round(r1,2),
                              round(r2,2),
                              round(r3,2)),
                              "Multi-layer Perceptron classifier\n feature to predict: {}".format(target_column),
                              "Numerical Multi-layer Perceptron\n feature to predict: {}\n used features: {}".format(target_column,keys)
                              ])
        
        print("All scores fall in the range 0-1")
        print("One being the best score on the three categories.")
        print(evaluation_table)
        print(evaluation_b)
        
    
    