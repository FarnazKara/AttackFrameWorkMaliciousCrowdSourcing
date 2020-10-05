# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:48:22 2018

@author: tahma
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:23:02 2018

@author: tahma
"""


import random
import pandas as pd
import numpy as np
#import collections 
from confusion_matrix import  Confusion_Matrix  as cm
from pathlib import Path, PureWindowsPath


class AttackStrategy(object):

    def __init__(self):
        
        self.num_labels = 2 
        self.infer_method =  ['mv','em','mvsoft','mvhard', 'bcc', 'lfc', 'kos', 'zc']
        self.accuracy_strategy = [0 for l in range(self.num_labels)] 
        num_adusers_percantage = [0, 5, 10, 20, 30, 40, 60, 70]
        
        self.update_acc_perIteration = [[0 for x in range(len(num_adusers_percantage))] for y in range(len(self.infer_method))]


    
    def initializeParameter(self, truth_labels,obfuscate):
        self.groundTruth = truth_labels
        #Percentage of tasks that attacker needs to behave like normal one
        # gamma == obfuscate hastesh 
        self.obfuscate = obfuscate
        self.accuracy_strategy = pd.DataFrame(columns = ['num_attackers', 'accuracy', 'method', 'iter'])  

    
    
    def appendAcctodf(self, acc_res):        
       result = self.accuracy_strategy.append(acc_res, ignore_index=True)
       self.accuracy_strategy = result.copy()
    
    def getAccuracy(self):
        return self.accuracy_strategy

    def saveAllAccuracy(self, filepath):
        self.accuracy_strategy.to_csv(filepath, sep=',', encoding='utf-8',index=False)

    def find(lst,a):
        return [i for i, x in enumerate(lst) if x == a]
    
    def getindextask(self,list1, percentage):
        n = int(len(list1) * percentage)
        partial = random.sample(list1,n)
        return partial
                
        
    def rand_chooseTask(self,mylist, n):
        """Return n tasks with truthlabel 0 or 1 ."""
        rnd_tasks = []
        for i in range(n):
            r = random.choice(mylist)
            while r in rnd_tasks:
                r = random.choice(mylist)#.randint(1, new_task)
            rnd_tasks.append(r)
        return rnd_tasks
    
    
    def common_elements(self,list1, list2):
        return list(set(list1).intersection(list2))
    
    
    def getlabelbasedCM(self, alpha, beta, truth):  
        if truth == 1.0 :
           if random.randrange(1,100) in range(1,int(alpha*100)):
              return 1
           else: return 0
        else:
            if random.randrange(1,100) in range(1,int(beta*100)):
                return 0
            else: 
                return 1        
    
    def CreateAttackData_increaseError(self,old_df,rnd_user,tasks_perUsers, file_path, 
                                       num_users, p_ad_id, iters, mean_a_adv,svd_a_adv,
                                       mean_b_adv, svd_b_adv, savingfilepath,g_honesttasks, confusion):  
        appended_df = old_df
        
        answer = pd.DataFrame(columns=['question','worker','answer'])
        num_tasks = old_df.shape[1]  
        
        adversaryConfusioMatrix = cm()
        adversaryConfusioMatrix.setCM(confusion)
        #adversaryConfusioMatrix.createCM(len(rnd_user), mean_a_adv,svd_a_adv,mean_b_adv, svd_b_adv)
        print("adversary setting")
        for idx, u_idx in enumerate(rnd_user):  
            normBehave_tidx = g_honesttasks[u_idx][self.obfuscate] #   self.getindextask(tasks_perUsers[u_idx],self.obfuscate)
            advBehave_tidx = set(tasks_perUsers[u_idx]).difference(normBehave_tidx)
            if len(advBehave_tidx) != 0:
                for ctask in advBehave_tidx:  
                    # I change to u_idx from idx
                    advLabel = self.getlabelbasedCM(adversaryConfusioMatrix.TTcdf[u_idx], adversaryConfusioMatrix.FFcdf[u_idx], self.groundTruth[ctask])
                    # it is Int I need to check for whole
                    if advLabel != appended_df.iloc[u_idx , ctask]:
                        #num_flip_att_tasks += 1
                        appended_df.iloc[u_idx , ctask] = advLabel
            else:
                print("Attackers behave like normal users")

        #self.set_avg_attackertasks(p_ad_id,num_att_tasks, num_flip_att_tasks,num_obf_tasks,num_lie_tasks, iters )
    
        num_line = 0
        for t in range(num_tasks):
           for u in range(num_users):
               if np.isnan(appended_df.loc[u][t]) == False: 
                   line = ['task_'+str(t), 'user_'+str(u), int(appended_df.loc[u][t])]
                   answer.loc[num_line] = line
                   num_line += 1
        
        answer.to_csv(file_path,sep=',', encoding='utf-8', index = False)        
        #answer.to_csv(savingfilepath , sep=',', encoding='utf-8',index = False)
        answer.to_csv(savingfilepath ,sep=',', encoding='utf-8',index = False)
        return appended_df 
 
    """
    # check if there is any common task that I have already infer rhe groundtruth or not:
            # Landa percent tell truth ad 1-landa tell lie
            # I have the reliability of thoes users tooo
    #else: 
             #do as before
    """
    


    def set_avg_attackertasks(self, p_ad_id,num_att_tasks, num_flip_att_tasks,num_obf_tasks,num_lie_task, iters):
        self.avg_num_att_tasks[p_ad_id]+= num_att_tasks/ iters
        self.avg_num_flip_att_tasks[p_ad_id] += num_flip_att_tasks /iters
        self.avg_num_lie_tasks[p_ad_id] += num_lie_task / iters
        self.avg_num_obf_tasks[p_ad_id] += num_obf_tasks /iters
    
    
    def get_avg_attackertasks(self):
        return  self.avg_num_att_tasks, self.avg_num_flip_att_tasks, self.avg_num_lie_tasks, self.avg_num_obf_tasks

    

    def update(self,acc, pidx):
        te = acc.groupby(['method','num_attackers']).mean()
        te = te.reset_index(level=['method', 'num_attackers'])
        for infM in self.infer_method:
            second = te[te['method'] == infM]['accuracy'].values[0]
            if infM == 'em':
                self.update_acc_perIteration[0] [pidx] += second
            elif infM == 'mv':
                self.update_acc_perIteration[1] [pidx] += second
            elif infM == 'mvhard':
                self.update_acc_perIteration[2] [pidx] += second
            elif infM == 'mvsoft':
                self.update_acc_perIteration[3] [pidx] += second
            elif infM == 'bcc':
                self.update_acc_perIteration[4] [pidx] += second
            elif infM == 'lfc':
                self.update_acc_perIteration[5] [pidx] += second
            elif infM == 'kos':
                self.update_acc_perIteration[6] [pidx] += second
            elif infM == 'zc':
                self.update_acc_perIteration[7] [pidx] += second


        return        


    def get_specific_Accuracy(self, infermethod, percentage_attacker_id):
        return self.update_acc_perIteration[infermethod][percentage_attacker_id]

