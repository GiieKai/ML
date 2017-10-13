
# coding: utf-8

# In[4]:

import csv
import numpy as np
import sys
from numpy.linalg import inv
import random
import math
data = []

for i in range(18):
    data.append([])
with open('train.csv', 'r', encoding='big5') as file:
    row = csv.reader(file , delimiter=",")
    n_row = 0
    for r in row :
        if n_row !=0:
            for i in range(3,27):
                
                if r[i] != "NR":
        
                    #y = 30%18，y = 12 
                    data[(n_row-1)%18].append(float(r[i]))
                else:
                    data[(n_row-1)%18].append(float(0))
        n_row = n_row + 1

train_x = []
train_y = []


ti = 9

thePower = 2
l_rate = 10000
repeat = 100000
for m  in range(12):
    for n in range(480-ti):
        #len(train_x) = 5652

        train_x.append([])
        for i in range(18):
 
            #if i in [2,4,5,6,7,8,9,12]:
            if i in [8,9]: 
            #if i ==9:
                for s in range(ti):

                    #for p in range(thePower):
                        #value = (data[i][480*m+n+s])**thePower
                        #train_x[(480-ti)*m+n].append(value)
                    train_x[(480-ti)*m+n].append(data[i][480*m+n+s])
            #else :
                #for s in range(ti):
                    #train_x[(480-ti)*m+n].append(float(0))

        train_y.append(data[9][480*m+n+ti])
        #data[9] 是 pm2.5
train_x = np.array(train_x)
train_y = np.array(train_y)
# add square term
if thePower >1:
    train_x = np.concatenate((train_x,train_x**(thePower)), axis=1)

train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x),axis = 1)
#init weight & other hyperparams
weight = np.ones((train_x.shape[1],1))
train_y= train_y.reshape(train_x.shape[0],1)
#lam = 0.1
s_gra = np.zeros((len(train_x[0]),1))
x_t = train_x.transpose()
for i in range(repeat):
    #l_rate = l_rate*0.1
    hypo = np.dot(train_x,weight)
    loss = hypo - train_y
    cost = np.sum(loss**2) / len(train_x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)#+lam*weight
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    weight = weight - l_rate*gra/ada
test_x = []
n_row = 0
test_input_file = sys.argv[1]
with open(test_input_file, 'r', encoding='big5') as file:
    row = csv.reader(file , delimiter= ",")
    for r in row:
        if n_row %18 == 0: 

            test_x.append([])
            #for i in range(11-ti,11):
                #test_x[n_row//18].append(float(r[i]) )
        else :
            #if n_row %18 in [2,4,5,6,7,8,9,12]: 
            if n_row %18 in [8,9]: 
            #if n_row %18 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
            #if n_row %18 ==9:
                for i in range(11-ti,11):
                    if r[i] !="NR":
                        test_x[n_row//18].append(float(r[i]))
                        #for p in range(thePower):
                            #value = (float(r[i]))**thePower
                            #test_x[n_row//18].append(value)
                    else:
                        test_x[n_row//18].append(0)
                        #for p in range(thePower):
                            #value = (float(0))**thePower
                            #test_x[n_row//18].append(value)                     
        n_row = n_row+1
test_x = np.array(test_x)
# add square term
if thePower >1:
    test_x = np.concatenate((test_x,test_x**(thePower)), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)    
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = float(np.dot(test_x[i],weight))
    ans[i].append(a)
#print(ans)
test_output_file = str(sys.argv[2])
text = open(test_output_file, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

