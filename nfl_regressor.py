# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
pd.options.plotting.backend = 'holoviews'
import hvplot
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time

class nflCombineRegressor:

    def __init__(self):
        snaps_cum_2013 = pd.Series(dtype = float)
        snaps_cum_2014 = pd.Series(dtype = float)
        snaps_cum_2015 = pd.Series(dtype = float)
        #snaps_cum_2016 = pd.Series(dtype = float)
        snaps_cum_2017 = pd.Series(dtype = float)

    def read_in(self,path): #change this to be relative via argparse()
        self.pd_2013 = pd.read_excel('/Users/aidankleinman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/DS 340W/DS-340W-Project/code/NFL 2013_edit.xlsx')
        self.pd_2014 = pd.read_excel('/Users/aidankleinman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/DS 340W/DS-340W-Project/code/NFL 2014_edit.xlsx')
        self.pd_2015 = pd.read_excel('/Users/aidankleinman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/DS 340W/DS-340W-Project/code/NFL 2015_edit.xlsx')
        self.pd_2016 = pd.read_excel('/Users/aidankleinman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/DS 340W/DS-340W-Project/code/NFL 2016_edit.xlsx')
        self.pd_2017 = pd.read_excel('/Users/aidankleinman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/DS 340W/DS-340W-Project/code/NFL 2017_edit.xlsx')
        
        self.snaps_2013 = pd.read_excel('/Users/aidankleinman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/DS 340W/DS-340W-Project/code/NFL 2013_edit.xlsx',
                                       sheet_name="Snaps")
        self.snaps_2014 = pd.read_excel('/Users/aidankleinman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/DS 340W/DS-340W-Project/code/NFL 2014_edit.xlsx',
                                       sheet_name="Snaps")
        self.snaps_2015 = pd.read_excel('/Users/aidankleinman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/DS 340W/DS-340W-Project/code/NFL 2015_edit.xlsx',
                                       sheet_name="Snaps")
        self.snaps_2016 = pd.read_excel('/Users/aidankleinman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/DS 340W/DS-340W-Project/code/NFL 2016_edit.xlsx', 
                                        sheet_name="Snaps")
        self.snaps_2017 = pd.read_excel('/Users/aidankleinman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/DS 340W/DS-340W-Project/code/NFL 2017_edit.xlsx',
                                       sheet_name="Snaps")

    def cumulative_snaps(self):
        """
        Sums all data across seasons,defense,offense, and special teams.
        This will change the the NANs to zeros
        """
        self.snaps_cum_2013 = self.snaps_2013.sum(axis = 1)
        self.snaps_cum_2014 = self.snaps_2014.sum(axis = 1)
        self.snaps_cum_2015 = self.snaps_2015.sum(axis = 1)
        self.snaps_cum_2016 = self.snaps_2016.sum(axis = 1)
        self.snaps_cum_2017 = self.snaps_2017.sum(axis = 1)
        
        print(len(self.snaps_cum_2013), "Samples started with - 2013")
        print(len(self.snaps_cum_2014), "Samples started with - 2014")
        print(len(self.snaps_cum_2015), "Samples started with - 2015")
        print(len(self.snaps_cum_2016), "Samples started with - 2016")
        print(len(self.snaps_cum_2017), "Samples started with - 2017")
    
    def split_test(self):
        index_nonzero_13 = self.snaps_cum_2013[self.snaps_cum_2013 !=0 ].index.tolist()
        index_nonzero_14 = self.snaps_cum_2014[self.snaps_cum_2014 !=0 ].index.tolist()
        index_nonzero_15 = self.snaps_cum_2015[self.snaps_cum_2015 !=0 ].index.tolist()
        index_nonzero_16 = self.snaps_cum_2016[self.snaps_cum_2016 !=0 ].index.tolist()
        index_nonzero_17 = self.snaps_cum_2017[self.snaps_cum_2017 !=0 ].index.tolist()

        snaps_parse_13 = self.snaps_cum_2013.iloc[index_nonzero_13]
        snaps_parse_14 = self.snaps_cum_2014.iloc[index_nonzero_14]
        snaps_parse_15 = self.snaps_cum_2015.iloc[index_nonzero_15]
        snaps_parse_16 = self.snaps_cum_2016.iloc[index_nonzero_16]
        snaps_parse_17 = self.snaps_cum_2017.iloc[index_nonzero_17]
        
        pd_2013_nozero = self.pd_2013.iloc[index_nonzero_13,:]
        pd_2014_nozero = self.pd_2014.iloc[index_nonzero_14,:]
        pd_2015_nozero = self.pd_2015.iloc[index_nonzero_15,:]
        pd_2016_nozero = self.pd_2016.iloc[index_nonzero_16,:]
        pd_2017_nozero = self.pd_2017.iloc[index_nonzero_17,:]
        
        cols =['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']
        
        x_data_13_ = pd_2013_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_14_ = pd_2014_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_15_ = pd_2015_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_16_ = pd_2016_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_17_ = pd_2017_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        index_nan_13 = x_data_13_.dropna().index.tolist()
        index_nan_14 = x_data_14_.dropna().index.tolist()
        index_nan_15 = x_data_15_.dropna().index.tolist()
        index_nan_16 = x_data_16_.dropna().index.tolist()
        index_nan_17 = x_data_17_.dropna().index.tolist()
    
        x_data_13_nonan = x_data_13_.loc[index_nan_13]
        x_data_14_nonan = x_data_14_.loc[index_nan_14]
        x_data_15_nonan = x_data_15_.loc[index_nan_15]
        x_data_16_nonan = x_data_16_.loc[index_nan_16]
        x_data_17_nonan = x_data_17_.loc[index_nan_17]
        
        y_data_13_nonan = snaps_parse_13.loc[index_nan_13]
        y_data_14_nonan = snaps_parse_14.loc[index_nan_14]
        y_data_15_nonan = snaps_parse_15.loc[index_nan_15]
        y_data_16_nonan = snaps_parse_16.loc[index_nan_16]
        y_data_17_nonan = snaps_parse_17.loc[index_nan_17]

        scaler = StandardScaler()
        x_data_13 = scaler.fit_transform(x_data_13_nonan)
        x_data_14 = scaler.fit_transform(x_data_14_nonan) 
        x_data_15 = scaler.fit_transform(x_data_15_nonan)
        x_data_16 = scaler.fit_transform(x_data_16_nonan)
        x_data_17 = scaler.fit_transform(x_data_17_nonan)
        
        df_13 = pd.DataFrame(x_data_13, columns = cols)
        df_14 = pd.DataFrame(x_data_14, columns = cols)
        df_15 = pd.DataFrame(x_data_15, columns = cols)
        df_16 = pd.DataFrame(x_data_16, columns = cols)
        df_17 = pd.DataFrame(x_data_17, columns = cols)
        
        x = pd.concat([df_13, df_14, df_15, df_17]) 
        y = pd.concat([y_data_13_nonan, y_data_14_nonan, y_data_15_nonan,
                        y_data_17_nonan])     
        
        print(len(x_data_13_nonan), "Samples started with - 2013")
        print(len(x_data_14_nonan), "Samples started with - 2014")
        print(len(x_data_15_nonan), "Samples started with - 2015")
        print(len(x_data_16_nonan), "Samples started with - 2016")
        print(len(x_data_17_nonan), "Samples started with - 2017")

        self.x_train,self.X_test,self.y_train,self.y_test = train_test_split(x,y)
     
    def model_test(self):
        self.model = GradientBoostingRegressor(learning_rate= 0.1, 
                                               n_estimators=110, max_depth = 4)
        self.model2 = RandomForestRegressor(n_estimators=100,min_weight_fraction_leaf=0.1)
        self.model3 = LinearRegression()
        self.model4 = DecisionTreeRegressor(min_samples_leaf=24)
        self.model5 = SVR(kernel = 'linear')
        
        self.model.fit(self.x_train,self.y_train)
        self.model2.fit(self.x_train,self.y_train)
        self.model3.fit(self.x_train,self.y_train)
        self.model4.fit(self.x_train,self.y_train)
        self.model5.fit(self.x_train,self.y_train)
 
        y_pred = self.model.predict(self.X_test)
        y_pred1 = self.model2.predict(self.X_test)
        y_pred2 = self.model3.predict(self.X_test)
        y_pred3 = self.model4.predict(self.X_test)
        y_pred4 = self.model5.predict(self.X_test)
        
        #Calculate error
        mse = mean_squared_error(self.y_test,y_pred,squared = False)
        mse1 = mean_squared_error(self.y_test,y_pred1,squared = False)
        mse2 = mean_squared_error(self.y_test,y_pred2,squared = False)
        mse3 = mean_squared_error(self.y_test,y_pred3,squared = False)
        mse4 = mean_squared_error(self.y_test,y_pred4,squared = False)
        # mse = r2_score(self.y_test, y_pred)
        # mse1 = r2_score(self.y_test, y_pred1)
        # mse2 = r2_score(self.y_test, y_pred2)
        # mse3 = r2_score(self.y_test, y_pred3)
        # mse4 = r2_score(self.y_test, y_pred4)
        return mse, mse1, mse2, mse3, mse4
        
    def plot_feature_importance(self):
        #Calculate feature importance 
        feature_imp = pd.Series(self.model.feature_importances_,index=self.X_test.columns).sort_values(ascending=False)
        feature_imp2 = pd.Series(self.model2.feature_importances_,index=self.X_test.columns).sort_values(ascending=False)
        feature_imp3 = pd.Series(self.model3.coef_[0],index=self.X_test.columns).sort_values(ascending=False)
        feature_imp4 = pd.Series(self.model4.feature_importances_,index=self.X_test.columns).sort_values(ascending=False)
        feature_imp5 = pd.Series(self.model5.coef_[0],index=self.X_test.columns).sort_values(ascending=False)
        fig, axs = plt.subplots(2, 3)
        axs = axs.flatten()
        sns.barplot(ax=axs[0],x=feature_imp,y=feature_imp.index)
        sns.barplot(ax=axs[1],x=feature_imp2,y=feature_imp2.index)
        sns.barplot(ax=axs[2],x=feature_imp4,y=feature_imp4.index)
        sns.barplot(ax=axs[3],x=feature_imp5,y=feature_imp5.index)
        sns.barplot(ax=axs[4],x=feature_imp3,y=feature_imp3.index)
        plt.xlabel('Feature Importance')
        axs[0].set_title('GradientBoostingRegressor')
        axs[1].set_title('RandomForestRegressor')
        axs[2].set_title('DecisionTreeRegressor')
        axs[3].set_title('SVC')
        axs[4].set_title('LinearRegression')
        plt.draw()
        plt.show()
        
        
        
if __name__ == '__main__':
    start_time = time.time()
    nfl = nflCombineRegressor()
    nfl.read_in("")
    nfl.cumulative_snaps()
    nfl.split_test()
    cols = ['Gradient_error', 'RFR_error', 'Linear_error','DT_error',
             'SVM_error']
    lst = []
    for i in range(0,20):
        save_list =  nfl.model_test()
        lst.append(save_list)
    error = pd.DataFrame(lst,columns=cols)
    nfl.plot_feature_importance()
    print("--- %s seconds ---" % (time.time() - start_time))
    hvplot.show(error.plot())
    


