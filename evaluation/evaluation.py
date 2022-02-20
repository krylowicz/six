from sdv.evaluation import evaluate 
from sdv.metrics.tabular import LogisticDetection,NumericalMLP,CSTest,KSTest,BinaryLogisticRegression,ContinuousKLDivergence,DiscreteKLDivergence
import numpy as np
import pandas as pd
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
                
    
    def table(self,target_column,cat_target,keys):        
        
        """
          target_colum = the column an attacker will try to predict
          cat_target= categorical target, it must be binary.
          keys = list of columns an attacker will use for predicting the target column, the key columns need to be the same object type
          as the target_column.
          
          Returns: Table of the scores for each evaluation category: resemblance, utility and privacy.
        """ 
        
        resemblance_1=KSTest.compute(self.real,self.synthetic)
        print("(1/7) Kolmogorov–Smirnov test...",resemblance_1)
        resemblance_2=CSTest.compute(self.real,self.synthetic)
        print("(2/7) Chi–Squared test...",resemblance_2)              
        resemblance_3=ContinuousKLDivergence.compute(self.real,self.synthetic)
        print("(3/7) ContinuousKLDivergence...",resemblance_3)
        resemblance_4=DiscreteKLDivergence.compute(self.real,self.synthetic)
        print("(4/7) DiscreteKLDivergence...",resemblance_4)
        resemblance_5=LogisticDetection.compute(self.real,self.synthetic)
        print("(5/7) Logistic Detection...",resemblance_5)
        
       
        utility=BinaryLogisticRegression.compute(self.real,self.synthetic,target=cat_target)
        print("(6/7) Binary Logistic Regression...",utility)

        privacy=NumericalMLP.compute(self.real, self.synthetic,sensitive_fields=[target_column],key_fields=keys)
        print("(7/7) Numerical Multi-Layer Perceptron...",privacy)
        
        df = pd.merge(self.real, self.synthetic, how='outer', indicator='Exist')
        duplicates_df = df.loc[df['Exist'] == 'both']
        duplicate_rows=duplicates_df.shape[0]
        
        evaluation_table=PrettyTable()
        evaluation_table.title="SCORES"          
        evaluation_table.field_names=["Resemblance", "Utility (TRTS)","Privacy","score_range","Goal","duplicate_rows"]
        evaluation_table.add_row([round((resemblance_1+resemblance_2+resemblance_3+resemblance_4+resemblance_5)/5,2),
                                  round(utility,2),round(privacy,2),"(0,1)","Maximize",duplicate_rows])          
        print(evaluation_table)
        