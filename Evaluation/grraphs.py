# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 14:14:31 2017

@author: tahma
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

"""
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
"""
def show_confusionMatrixRate(truth_type, len_adversaries, iterations, df ):
    test_perf = df[df['method']== truth_type]
    test_perf = test_perf[['FNR','FPR','TNR','TPR']]
    end = len_adversaries # len(num_adusers_percantage)       
    for r in range(iterations):
        test_perf[r * end :end *(r+1) -1][:].plot()

def effectnumberUsers(users, accuracy, title):
    plt.figure()
    plt.plot(users, accuracy, 'ro')
    plt.title(title)
    plt.xlabel('users')
    plt.ylabel('accuracy')
    plt.show()
    return

def plotTwoList(xaxies, yaxies_m1, yaxies_m2, variedIndex, title):
    plt.figure()
    plt.title(title)
    plt.xlabel(variedIndex)
    plt.ylabel('accuracy')
    plt.plot(xaxies,yaxies_m1,'g')
    plt.plot(xaxies,yaxies_m2,'r')
    plt.show()
    return
#        



def effectnumberTasks(userCount, attackerPercentage):
    return

    
def accuracy_charts(ch_title, type_accuracy_strategy,  len_adusers_percantage, nusers,percentage_attackers, typechart, uniquelabels, 
                    counts_label, workermodel, savingpath):
    all_accuracies = pd.DataFrame()
    all_accuracies['nadversary'] = [int(nusers*x/100) for x in percentage_attackers]
    te = type_accuracy_strategy.groupby(['method','num_attackers']).mean()
    models = ['bcc','em','mv','mvhard', 'mvsoft'] #based on alphabet
    for idx, val in enumerate(models): 
        start =  len_adusers_percantage * idx
        end = len_adusers_percantage * (idx +1)
        all_accuracies[val] = te[typechart][start:end].values
    plt.figure()
    ax = all_accuracies.plot(title = ch_title,  x='nadversary',  figsize=(10, 10) )
    ax.set_xlabel("Number of attackers",size = 12,color="r",alpha=0.5)
    ax.set_ylabel(typechart,size = 12,color="r",alpha=0.5)
    #TODO: correct for all 
    sectick = max(all_accuracies['mvsoft'].append(all_accuracies['bcc']).reset_index(drop=True))+0.1
    tick =max( sectick,  max(all_accuracies['em'].append(all_accuracies['mv']).reset_index(drop=True))+0.1)
    ax.set_yticks(np.arange(0, tick, 0.1))
    #txtinf = " ".join(['num_user=', str(nusers),workermodel ,' trueLabel[0,1] =', str(counts_label[0]), str(counts_label[1])])
    #ax.text(0.5, 0.5, txtinf)
    plt.savefig(savingpath)
    plt.show()
    del all_accuracies['nadversary']

def violongraph(data):
    model_df = data.copy()
    mv_df = pd.DataFrame(columns= ['att_0','att_5','att_10','att_20','att_40','att_60'])
    model_df = data[data['method'] == 'mv'].reset_index()
    mv_df['att_0'] = model_df[model_df['num_attackers']== 0]['accuracy'].reset_index(drop= True)
    mv_df['att_5'] = model_df[model_df['num_attackers']== 5]['accuracy'].reset_index(drop= True)
    mv_df['att_10'] = model_df[model_df['num_attackers']== 10]['accuracy'].reset_index(drop= True)
    mv_df['att_20'] = model_df[model_df['num_attackers']== 20]['accuracy'].reset_index(drop= True)
    mv_df['att_40'] = model_df[model_df['num_attackers']== 40]['accuracy'].reset_index(drop= True)
    mv_df['att_60'] = model_df[model_df['num_attackers']== 60]['accuracy'].reset_index(drop= True)

    f, ax = plt.subplots(figsize=(12, 12))
    sns.violinplot(data=mv_df)
    sns.despine(offset = 10, trim=True)


def avg_task_attackers(attacker_percentage, avg_tasks, ylabel, title, savingpath):
    plt.figure()
    #plt.plot(attacker_percentage, avg_tasks[0], 'r--',  label='label 0')
    #plt.plot(attacker_percentage, avg_tasks[1], 'bs', label='label 1')
    plt.plot(attacker_percentage, avg_tasks, 'b-', label='label 1')
    plt.xlabel("attackers (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(savingpath)
    plt.show()

    

def getUserTask(num_users, cnt_tasks):
    plt.scatter(num_users, cnt_tasks, c = 'r',s= 100, marker = '+')
    plt.xticks(num_users)
    plt.yticks(cnt_tasks)
    plt.xlabel("number_of_users")
    plt.ylabel("number_of_tasks")
    plt.show()
    
    rate = [num_users[i] / cnt_tasks[i] for i in range(len(num_users))]
    plt.scatter(num_users, rate)
    plt.xticks(num_users)
    plt.xlabel("number_of_users")
    plt.ylabel("rate_user per task")
    plt.show()
"""
    
def getpercision(y_true, y_pred):
     return  precision_score(y_true, y_pred, average='micro')  

def getFscore(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")
    #return precision_recall_fscore_support(y_true, y_pred, average='micro')

def plotPercision_recall(y_test, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    average_precision = average_precision_score(y_test, y_score)
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))  




def accuracy_charts(ch_title, type_accuracy_strategy,  len_adusers_percantage, nusers,percentage_attackers, typechart):
    all_accuracies = pd.DataFrame()
    all_accuracies['nadversary'] = [int(nusers*x/100) for x in percentage_attackers]
    te = type_accuracy_strategy.groupby(['method','num_attackers']).mean()
    models = ['em','mv']
    for idx, val in enumerate(models): 
        start =  len_adusers_percantage * idx
        end = len_adusers_percantage * (idx +1)
        all_accuracies[val] = te[typechart][start:end].values
    ax = all_accuracies.plot(title = ch_title,  x='nadversary')
    ax.set_xlabel("Number of attackers",size = 12,color="r",alpha=0.5)
    ax.set_ylabel(typechart,size = 12,color="r",alpha=0.5)
    del all_accuracies['nadversary']
    #violongraph(all_accuracies) 

"""