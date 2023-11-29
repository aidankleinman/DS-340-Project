#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nfl_regressor import nflCombineRegressor
import pandas as pd
import numpy as np
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.inspection import permutation_importance
import hvplot


class nflCombineClassify(nflCombineRegressor):
    
    def __init__(self,path):
        super().__init__()
        super().read_in(path) #change this to be relative via argparse()
        super().cumulative_snaps()
        
    def snaps_to_binary(self):
        cols =['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']
        
        x_data_13_ = self.pd_2013[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_14_ = self.pd_2014[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_15_ = self.pd_2015[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        #x_data_16_ = self.pd_2015[['40yd','Vertical','BP','Broad Jump',
                          #'Shuttle','3Cone']]
        #x_data_17_ = self.pd_2017[['40yd','Vertical','BP','Broad Jump',
                          #'Shuttle','3Cone']]

        index_nan_13 = x_data_13_.dropna().index.tolist()
        index_nan_14 = x_data_14_.dropna().index.tolist()
        index_nan_15 = x_data_15_.dropna().index.tolist()
        #index_nan_16 = x_data_16_.dropna().index.tolist()
        #index_nan_17 = x_data_17_.dropna().index.tolist()

        y_data_13_nonan = self.snaps_cum_2013.loc[index_nan_13]
        y_data_14_nonan = self.snaps_cum_2014.loc[index_nan_14]
        y_data_15_nonan = self.snaps_cum_2015.loc[index_nan_15]
        #y_data_16_nonan = self.snaps_cum_2016.loc[index_nan_16]
        #y_data_17_nonan = self.snaps_cum_2017.loc[index_nan_17]
        
        x_data_13_nonan = x_data_13_.loc[index_nan_13]
        x_data_14_nonan = x_data_14_.loc[index_nan_14]
        x_data_15_nonan = x_data_15_.loc[index_nan_15]
        #x_data_16_nonan = x_data_16_.loc[index_nan_16]
        #x_data_17_nonan = x_data_17_.loc[index_nan_17]
        
        print(len(y_data_13_nonan), "Samples ended with - 2013")
        print(len(y_data_14_nonan), "Samples ended with - 2014")
        print(len(y_data_15_nonan), "Samples ended with - 2015")
        #print(len(y_data_16_nonan), "Samples ended with - 2016")
        #print(len(y_data_17_nonan), "Samples ended with - 2017")
        
        #convert to binary
        y_data_13_nonan[y_data_13_nonan > 0] = 1
        y_data_14_nonan[y_data_14_nonan > 0] = 1
        y_data_15_nonan[y_data_15_nonan > 0] = 1
        #y_data_16_nonan[y_data_16_nonan > 0] = 1
        #y_data_17_nonan[y_data_17_nonan > 0] = 1
        
        y = pd.concat([y_data_13_nonan, y_data_14_nonan, y_data_15_nonan]).astype(int)
        x = pd.concat([x_data_13_nonan, x_data_14_nonan, x_data_15_nonan])
        #split the data into data to train the model on and unseen data to test the model for a final assessment
        self.x_train_classify,self.X_test_classify,self.y_train_classify,self.y_test_classify = train_test_split(x,y,train_size=0.75,test_size=0.25)
        print("Training data size: ",self.x_train_classify.shape[0])
        print("Testing data size: ", self.X_test_classify.shape[0])

    #def split_train_validation(self, validation_size=0.50):
        # Split the current training data into a smaller training set and a validation set for hyperparameter tuning
        #self.x_train_smaller, self.x_validation, self.y_train_smaller, self.y_validation = \
            #train_test_split(self.x_train_classify, self.y_train_classify, test_size=validation_size, random_state=42)
        #print("small Training data size: ",self.x_train_smaller.shape[0])
        #print("validation data size: ", self.x_validation.shape[0])

    def feature_selection(self, k = 5):
        self.feature_selector = SelectKBest(score_func = chi2, k=k)
        self.kX_train_classify = self.feature_selector.fit_transform(self.x_train_classify, self.y_train_classify)
        #self.kX_validation = self.feature_selector.transform(self.x_validation)
        self.kX_test_classify = self.feature_selector.transform(self.X_test_classify)
    def model_test_classify(self):
        
        self.model1_classify = DecisionTreeClassifier(criterion='entropy')
        self.model2_classify = GradientBoostingClassifier(n_estimators=105,max_depth=4,tol=0.001)
        self.model3_classify = SVC(kernel='linear')
        self.model4_classify = GaussianNB()
        self.model5_classify = RandomForestClassifier(n_estimators=105,criterion='entropy',min_samples_leaf=4)
        self.model6_classify = LogisticRegression(max_iter=200)
        self.model7_classify = KNeighborsClassifier(n_neighbors=5)
        self.model1_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model2_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model3_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model4_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model5_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model6_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model7_classify.fit(self.kX_train_classify,self.y_train_classify)

        
        y_pred1 = self.model1_classify.predict(self.X_test_classify)
        y_pred2 = self.model2_classify.predict(self.X_test_classify)
        y_pred3 = self.model3_classify.predict(self.X_test_classify)
        y_pred4 = self.model4_classify.predict(self.X_test_classify)
        y_pred5 = self.model5_classify.predict(self.X_test_classify)
        y_pred6 = self.model6_classify.predict(self.X_test_classify)
        y_pred7 = self.model7_classify.predict(self.kX_test_classify)

        
        print("DecisionTreeClassifier Validation Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred1))
        print("GradientBoostingClassifier Validation Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred2))
        print("SVC Validation Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred3))
        print("GaussianNB Validation Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred4))
        print("RandomForestClassifier Validation Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred5))
        print("LogisticRegression Validation Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred6))
        print("KNeighbors Classifier Validation Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred7))

    def model_evaluation(self, k=5):
        x_data_17_ = self.pd_2017[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        index_nan_17 = x_data_17_.dropna().index.tolist()
        y_data_17_nonan = self.snaps_cum_2017.loc[index_nan_17]
        x_data_17_nonan = x_data_17_.loc[index_nan_17]
        y = pd.concat([y_data_17_nonan]).astype(int)
        x = pd.concat([x_data_17_nonan])

        self.x_train_classify,self.X_test_classify,self.y_train_classify,self.y_test_classify = train_test_split(x,y,train_size=0.65,test_size=0.35)
        #self.feature_selector = SelectKBest(score_func = chi2, k=k)
        #self.kX_train_classify = self.feature_selector.fit_transform(self.x_train_classify, self.y_train_classify)
        self.kX_test_classify = self.feature_selector.transform(self.X_test_classify)

        y_pred1 = self.model1_classify.predict(self.X_test_classify)
        y_pred2 = self.model2_classify.predict(self.X_test_classify)
        y_pred3 = self.model3_classify.predict(self.X_test_classify)
        y_pred4 = self.model4_classify.predict(self.X_test_classify)
        y_pred5 = self.model5_classify.predict(self.X_test_classify)
        y_pred6 = self.model6_classify.predict(self.X_test_classify)
        y_pred7 = self.model7_classify.predict(self.kX_test_classify)

        #print("Unique values in y_test_classify:", np.unique(self.y_test_classify))
        #print("Unique values in y_pred1:", np.unique(y_pred1))
        #print("Unique values in y_pred2:", np.unique(y_pred2))
        #print("Unique values in y_pred3:", np.unique(y_pred3))
        #print("Unique values in y_pred4:", np.unique(y_pred4))
        #print("Unique values in y_pred5:", np.unique(y_pred5))
        #print("Unique values in y_pred6:", np.unique(y_pred6))
        #print("Unique values in y_pred7:", np.unique(y_pred7))

        # Now check the format of the targets and predictions
        #print("y_test_classify shape:", self.y_test_classify.shape)
        #print("y_pred1 shape:", y_pred1.shape)
        #print("y_pred2 shape:", y_pred2.shape)
        #print("y_pred3 shape:", y_pred3.shape)
        #print("y_pred4 shape:", y_pred4.shape)
        #print("y_pred5 shape:", y_pred5.shape)
        #print("y_pred6 shape:", y_pred6.shape)
        #print("y_pred7 shape:", y_pred7.shape)

        # Example snippet to verify X_test_classify and y_test_classify assignments
        #print("Shape of X_test_classify:", self.X_test_classify.shape)
        #print("Unique values in X_test_classify:", np.unique(self.X_test_classify))
        #print("Shape of y_test_classify:", self.y_test_classify.shape)
        #print("Unique values in y_test_classify:", np.unique(self.y_test_classify))


        print("DecisionTreeClassifier Testing Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred1))
        print("GradientBoostingClassifier Testing Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred2))
        print("SVC Testing Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred3))
        print("GaussianNB Testing Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred4))
        print("RandomForestClassifier Testing Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred5))
        print("LogisticRegression Testing Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred6))
        print("KNeighbors Classifier Testing Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred7))

    
    def plot_feature_importance_classify(self,save_path=None):
        imps = permutation_importance(self.model4_classify, self.X_test_classify, self.y_test_classify)
        # Calculate feature importance 
        feature_imp1 = pd.Series(self.model1_classify.feature_importances_, index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp2 = pd.Series(self.model2_classify.feature_importances_, index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp4 = pd.Series(imps.importances_mean, index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp3 = pd.Series(self.model3_classify.coef_[0], index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp5 = pd.Series(self.model5_classify.feature_importances_, index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp6 = pd.Series(self.model6_classify.coef_[0], index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp7 = pd.Series(self.feature_selector.scores_, index=self.X_test_classify.columns).sort_values(ascending=False)

        fig, axs = plt.subplots(2, 4, figsize=(15, 10))  # Set the number of subplots to 7 and adjust the figure size
        axs = axs.flatten()

        sns.barplot(x=feature_imp1, y=feature_imp1.index, ax=axs[0])
        sns.barplot(x=feature_imp2, y=feature_imp2.index, ax=axs[1])
        sns.barplot(x=feature_imp3, y=feature_imp3.index, ax=axs[2])
        sns.barplot(x=feature_imp4, y=feature_imp4.index, ax=axs[3])
        sns.barplot(x=feature_imp5, y=feature_imp5.index, ax=axs[4])
        sns.barplot(x=feature_imp6, y=feature_imp6.index, ax=axs[5])
        sns.barplot(x=feature_imp7, y=feature_imp7.index, ax=axs[6])
        
        for ax in axs:
            ax.set_xlabel('Feature Importance')  # Set the same x-label for all subplots
        plt.suptitle('Feature Importance by Model', fontsize=16)  # Universal title
        
        axs[0].set_title('DecisionTreeClassifier')
        axs[1].set_title('GradientBoostingClassifier')
        axs[2].set_title('SVC')
        axs[3].set_title('GaussianNB')
        axs[4].set_title('RandomForestClassifier')
        axs[5].set_title('LogisticRegression')
        axs[6].set_title("KNeighborsClassifier")
        plt.delaxes(axs[7])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to accommodate the title

        # Save the plot as a PNG image
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')

        plt.show()

            
if __name__ == '__main__':
    classify = nflCombineClassify('')
    classify.snaps_to_binary()
    #classify.split_train_validation(0.2)
    classify.feature_selection(k=5)
    classify.model_test_classify()
    classify.model_evaluation()
    lst = []
    cols = ['acc']
    # h_para = 100
    # for i in range(0,20):
    #     save_list =  classify.model_test_classify(h_para)
    #     lst.append(save_list)
    #     h_para =+ 5 
    # acc = pd.DataFrame(lst,columns=cols)
    # hvplot.show(acc.plot())
    classify.plot_feature_importance_classify(save_path='./plots/feature_importance_plot.png')
