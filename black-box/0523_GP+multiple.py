
# coding: utf-8

# In[1]:


import numpy as np
import operator
import random
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn as sk

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error
#快速1次轉2次
from sklearn import preprocessing 
from sklearn.preprocessing import PolynomialFeatures

from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RationalQuadratic,ConstantKernel


# In[31]:


#建立GP模型

#先把最一開始的資料Excel叫出來
path =r'./data/raw_data/'
files = os.listdir(path)
raw_Y = {}
for f in files:
    #print(f)
    df = pd.DataFrame()
    df = pd.read_excel(path+f)
    #print(df)
    YY = np.zeros(810)
    for i in range(810):
        YY[i] = df['y_value'][i]
    raw_Y[f] = YY.reshape(810,1)
X = np.zeros((810,5))
for col in range(0,5):
    temp = []
    temp = df[col][0:810]
    for i in range(810):
        X[i][col] = temp[i]
raw_X = np.array(X)

#print(raw_X.shape ,raw_Y['vel_3_data.xlsx'].shape)
#(810, 5) (810, 1)



#設定 Kernel 的起始參數
kernel_Gp = ConstantKernel(
    constant_value=1.0,constant_value_bounds=(0.1,10.0)) * RationalQuadratic(
    alpha=0.5,length_scale=3,alpha_bounds=(0.1,10),length_scale_bounds=(0.1,10))

#宣告模型設定
model_vel_03 = gpr(kernel=kernel_Gp,n_restarts_optimizer=5,normalize_y="False",optimizer='fmin_l_bfgs_b',random_state=None);
model_vel_15 = gpr(kernel=kernel_Gp,n_restarts_optimizer=5,normalize_y="False",optimizer='fmin_l_bfgs_b',random_state=None);
model_TI_03 = gpr(kernel=kernel_Gp,n_restarts_optimizer=5,normalize_y="False",optimizer='fmin_l_bfgs_b',random_state=None);
model_TI_15 = gpr(kernel=kernel_Gp,n_restarts_optimizer=5,normalize_y="False",optimizer='fmin_l_bfgs_b',random_state=None);

#開始訓練模型 
model_vel_03.fit(X=raw_X,y=raw_Y['vel_3_data.xlsx'])
model_vel_15.fit(X=raw_X,y=raw_Y['vel_15_data.xlsx'])
model_TI_03.fit(X=raw_X,y=raw_Y['TI_3_data.xlsx'])
model_TI_15.fit(X=raw_X,y=raw_Y['TI_15_data.xlsx'])

#存取模型
f = open('model_vel_03', 'wb')
pickle.dump(model_vel_03, f)
f = open('model_vel_15', 'wb')
pickle.dump(model_vel_15, f)
f = open('model_TI_03', 'wb')
pickle.dump(model_TI_03, f)
f = open('model_TI_15', 'wb')
pickle.dump(model_TI_15, f)


# In[40]:


#建立 複回歸模型

#load training data
raw_data = pd.read_excel('data/all_raw_data.xlsx')

raw_data_Y = raw_data[["y_value_TI_3","y_value_TI_15","y_value_vel_3","y_value_vel_15"]]

Y_list = ["y_value_TI_3","y_value_TI_15","y_value_vel_3","y_value_vel_15"]

X = np.load('data/X_data.npy')
#81 case
#(81, 4)

#deg 代表我要將我的模型設幾次方
#利用dic 存取模型
def train_model(inX,deg = 2):
    poly = PolynomialFeatures(deg,include_bias = False)
    TrainX = poly.fit_transform(inX)
    pre_fun = {}
    mape_list = {}
    R_list = {}
    for index,y in enumerate (Y_list):
        raw_Y = raw_data_Y[y]
        for i in range(10):
            model = linear_model.LinearRegression()
            a = range(i,810,10)
            Train_Y = raw_Y[list(a)]
            pre_fun[y[8:]+"_"+str(i)] = model.fit(TrainX,Train_Y)
            mape_list[y[8:]+"_"+str(i)] = mean_absolute_error(Train_Y,model.predict(TrainX))
            
    return pre_fun,mape_list

Multiple_Regression_Model ,MAE = train_model(X)


# In[52]:


# 預測
def input_data_process(para = X, d  = np.array( [[ 1.2, 2 , 3, 4, 5, 6, 7, 8, 9, 10 ] ]) ):
    input_data = d.transpose()
    for i in para:
        temp = [[i]]*10
        input_data = np.concatenate((input_data, temp), axis=1)
    return input_data

def pred(model,inD,deg = 2):
    if model == 'GP':        
        X = input_data_process(inD)
        vel_03_pred = model_vel_03.predict(X)
        vel_15_pred = model_vel_15.predict(X)
        TI_03_pred  =  model_TI_03.predict(X)
        TI_15_pred  =  model_TI_15.predict(X)
        predict = np.concatenate((TI_03_pred, TI_15_pred, vel_03_pred, vel_15_pred),axis= 1)
        predict = predict.T
        
        return predict.reshape(1,40)
    else:
        poly = PolynomialFeatures(deg,include_bias = False)
        Pre_X = poly.fit_transform([inD])
        pre_list = []
        for index,y in enumerate (Y_list):
            for i in range(10):
                pre_list.append(model[y[8:]+"_"+str(i)].predict(Pre_X))
        #print (pre_list)
        #"y_value_TI_3","y_value_TI_15","y_value_vel_3","y_value_vel_15"
        LL = np.array(pre_list)
        L_T = LL.T
        
        return L_T 


# In[441]:


#load exp_data
#"y_value_TI_3","y_value_TI_15","y_value_vel_3","y_value_vel_15"

exp = np.load("data/exp_data")
exp_pd = pd.DataFrame(data=exp)
exp_pd = exp_pd[['TI_3_data','TI_15_data','vel_3_data','vel_15_data']]
exp_data = np.array(exp_pd)
exp_cal_data = exp_data.reshape(1,40)

exp_T = exp_data.T

exp_cal_data_T = exp_T.reshape(1,40)


# In[443]:


# 建立 我們要的 目標函式

def objective_function( model,v=[0.5,2,0.5,2]):
    
    result_data = pred(model,v)
    
    #MAPE
    return np.mean(np.abs((result_data- exp_cal_data_T)/exp_cal_data_T))
    
    #RMSPE
    #return np.sqrt(np.mean(np.square((result_data - exp_cal_data_T)/exp_cal_data_T), axis=1,dtype=np.float64))[0]
    #RMSE
    #return np.sqrt(np.mean(np.square((result_data - exp_cal_data_T)), axis=1,dtype=np.float64))[0]
    
    
def positive_or_negative():
    if random.random() < 0.5:
        return 1
    else:
        return -1

def randam_h(h = 1):
    #h = 0.001
    a = positive_or_negative()*random.uniform(0,h)
    b = positive_or_negative()*random.uniform(0,(h-a**2)**0.5)
    c = positive_or_negative()*random.uniform(0,(h-a**2-b**2)**0.5)
    d = positive_or_negative()*(h**2-a**2-b**2-c**2)**0.5
    #print((a**2+b**2+c**2+d**2)**0.5)
    rand = [a,b,c,d]
    random.shuffle(rand)
    return rand

global lower_range 
lower_range = [0.01, 0.01, 0.01, 0.01]
global upper_range
upper_range =[1,4,1,4]

def drect_search(model,theta_0 = [0.5,2,0.5,2],dis = 0.01,points=10,tolerance_value = 0.15,Iteration_max=30,fast = 'adagrad' ):
    value = objective_function(model,v = theta_0)
    Iteration = 0
    initial_step_size = 0.01
    error = 0.0001
    s_gra = np.array([error]*4)
    value_history = []
    theta_history = []
    last_history = []    
    
    
    while tolerance_value < value or Iteration < Iteration_max:
        
        h = np.array([randam_h(dis) for i in range(points)])
        next_thetas = theta_0+h
        #print(next_thetas)
        #做邊界的判斷 
        for i,current_list in enumerate(next_thetas):
            u=[current if current >= lower and current <=upper
               else lower if current < lower else upper 
               for current, lower, upper in zip(current_list, lower_range, upper_range)]
            next_thetas[i] = u
            
        next_theta_temp = min(next_thetas,key=lambda item: objective_function(model,item))
        
        
        if value < objective_function(model,next_theta_temp):
            next_theta_temp = theta_0.copy()
            #dis = dis/2
        gradient = next_theta_temp - theta_0
#         print(type(gradient))
#         print(gradient.real)
        if fast == 'adagrad' :
            #print(fast)
            #加速收斂
            #print(s_gra)
            s_gra += [ x**2 for x in gradient.real ]

            ada = np.sqrt(s_gra)        

            step_size = initial_step_size/ada        


            next_theta = theta_0 + step_size*gradient
            #print(gradient)
        else :
            step_size = 1
            next_theta = theta_0 + step_size*gradient

        next_value = objective_function(model,next_theta)
        
        
        theta_0,value = next_theta,next_value
        
        tolerance_value = value        
        
        Iteration +=1
        
        
        value_history.append([Iteration,value, "tolerance_value",theta_0])
        
        theta_history.append(theta_0)
        #print(plot_history)
        
    last_history.append([Iteration,value, "tolerance_value",theta_0])
    
    return value_history,last_history


# In[27]:


#see the performance without ada
value,last = drect_search(Multiple_Regression_Model,[0.05,2.15,0.65,0.05],dis = 0.01,fast='no',Iteration_max=50)
plt.plot([i[0] for i in value],[i[1] for i in value],label = 'adagrad')
value_,last = drect_search(Multiple_Regression_Model,[0.05,2.15,0.65,0.05],dis = 0.01,Iteration_max=50)
plt.plot([i[0] for i in value_],[i[1] for i in value_],label = 'without adagrad')

plt.ylabel("MAPE")
plt.xlabel("Iteration")
plt.legend(loc='best')
print(last[0][1])


# In[449]:


#see the performance with ada
value,last = drect_search(Multiple_Regression_Model,[0.05,2.15,0.65,0.05],dis = 0.01,Iteration_max=50)
plt.plot([i[0] for i in value],[i[1] for i in value])
print(last[0][1])


# In[64]:


pred('GP',[0.1,1.3,0.4,2.0])


# In[65]:


exp_cal_data_T


# In[144]:


np.mean(np.abs((pred('GP',[0.1,1.3,0.4,2.0]) - exp_cal_data_T)/exp_cal_data_T))


# In[139]:


(pred('GP',[0.01,2.13,0.6345,0.1]) - exp_cal_data_T)/exp_cal_data_T


# In[140]:


np.abs((pred('GP',[0.01,2.13,0.6345,0.1]) - exp_cal_data_T)/exp_cal_data_T)


# In[150]:


np.mean(np.abs((pred('GP',[0.01,2.13,0.6345,0.1]) - exp_cal_data_T)/exp_cal_data_T))


# In[154]:


objective_function('GP',[0.01,2.13,0.6345,0.1])


# In[151]:


objective_function(Multiple_Regression_Model,[0.01,2.13,0.6345,0.1])

