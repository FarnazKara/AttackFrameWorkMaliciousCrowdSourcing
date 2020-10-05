# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:41:48 2018

@author: tahma
"""
import math 
import numpy as np
import random , csv
import pandas as pd
import truth_methods as tm 
from collections import defaultdict
from confusion_matrix import  Confusion_Matrix  as cm



class AttackStrategy:
    def __init__(self,num_adusers_percantage, infer_method, truthfile):
        self.accuracy_strategy = pd.DataFrame(columns = ['num_attackers', 'accuracy', 'method', 'iter'])  
        self.update_acc_perIteration = [[0 for x in range(len(num_adusers_percantage))] for y in range(len(infer_method))]
        self.truthfile = truthfile
        self.infermethod = infer_method
        self.num_adusers_percantage = num_adusers_percantage
        return

    def settingparameters(self, datadir, truthfile, ansfile, graph_dir,
                 mean_a_adv,mean_b_adv, mean_a_norm,mean_b_norm):
        self.datadir = datadir
        self.graphdir = graph_dir
        self.answerfile = ansfile
        
        self.tasksList = self.readfilegetcol(truthfile, 'question')
        self.honest_users_List = self.readfilegetcol(ansfile,'worker')
        self.ntasks = len(self.tasksList)
        
        self.numhonestUsers = len(self.honest_users_List )
        self.avg_adv_alph = mean_a_adv
        self.avg_adv_beta = mean_b_adv
        self.avg_norm_alph = mean_a_norm
        self.avg_norm_beta = mean_b_norm
        #self.num_adusers_percantage = [0,5,10,20,40,60]

        
        df = pd.read_csv(self.truthfile)
        self.true_labels = df['truth']

        
    def gettaskName(self, idx):
        return self.tasksList[idx]
    
    def getAccuracy(self):
        return self.accuracy_strategy

    
    def getMultiplyNums(self, value):
        N = math.ceil(math.sqrt(value))        
        while value % N != 0:
                N +=1
        M = value / N 
        return N, M
            
    
    def assignAttackerToTask(self,num_adversary, mean, ntasks):
            shape, scale = self.getMultiplyNums(mean)
            assigned = np.random.gamma(shape, scale, (num_adversary + self.numhonestUsers))
            num_assigned = random.sample(list(assigned),num_adversary)
            taskperadversary = {}
            for i, m in enumerate(num_assigned):
                tasksID = random.sample(range(0, ntasks), int(m))
                adversary_id = 'adv_{}'.format(i)
                taskperadversary[adversary_id] = tasksID
            
            return taskperadversary
        
        
    def getlabelbasedCM(self, alpha, beta, truth):  
            #print("truth is:")
            #print(truth)
            if truth == 1.0:
               if random.randrange(1,100) in range(1,int(alpha*100)):
                  return 1
               else: return 0
            else:
                if random.randrange(1,100) in range(1,int(beta*100)):
                    return 0
                else: 
                    return 1        
    
        
    def assignLabel(self,adv, adversayReliability_alpha, adversayReliability_beta, assignedTasks, normBehave_tidx, advBehave_tidx, norm_alpha, norm_beta ):
            truth = self.true_labels    
            line = []
            if len(advBehave_tidx) != 0:
                for ctask in advBehave_tidx:
                         label = self.getlabelbasedCM(adversayReliability_alpha, adversayReliability_beta ,truth[ctask] )
                         task_name = self.gettaskName(ctask)
                         line.append([task_name ,adv,  label])
            for ctask in normBehave_tidx:
                #print("Attackers behave like normal users")               
                label = self.getlabelbasedCM(norm_alpha, norm_beta ,truth[ctask] )
                task_name = self.gettaskName(ctask)
                line.append([task_name, adv, label])
            """
            with open("output.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(line)
            """
            return line
        
    def assign_adversary_reliability(self, numAttacker):
        adversaryConfusioMatrix = cm()
        adversaryConfusioMatrix.createCM(numAttacker, self.avg_adv_alph,0.002,self.avg_adv_beta, 0.002)
        
        return adversaryConfusioMatrix
    
    def assign_Normal_reliability(self, numAttacker):
        adversaryConfusioMatrix = cm()
        adversaryConfusioMatrix.createCM(numAttacker, self.avg_norm_alph,0.002,self.avg_norm_beta, 0.002)
        
        return adversaryConfusioMatrix


    
    def getindextask(self,list1, percentage):
        n = int(len(list1) * percentage)
        partial = random.sample(list1,n)
        return partial

    
    
    def poisionAttack(self, num_adversary, assignedTask, obfuscate):
            advAnswers = []
            adversayReliability = self.assign_adversary_reliability(num_adversary)
            norm_adversay_behave = self.assign_Normal_reliability(num_adversary)

            for i in range(num_adversary):
                idx = 'adv_{}'.format(i)
                normBehave_tidx = self.getindextask(assignedTask[idx],obfuscate)
                advBehave_tidx = set(assignedTask[idx]).difference(normBehave_tidx)
                
                advAnswers += self.assignLabel(idx, adversayReliability.TTcdf[i],
                                               adversayReliability.FFcdf[i], assignedTask[idx],
                                               normBehave_tidx, advBehave_tidx,
                                               norm_adversay_behave.TTcdf[i],
                                               norm_adversay_behave.FFcdf[i])
                
            return advAnswers
            
    def mergeTwoCSVFile(self, first, second, savingpath):
            
            results = pd.DataFrame([])
            namedf = pd.read_csv(second, skiprows=0, usecols=[0,1,2])
            results = results.append(namedf)

            namedf = pd.read_csv(first, usecols=[0,1,2])
            results = results.append(namedf)
            
             
            results.to_csv(savingpath)
            return results
        


    def appendAcctodf(self, acc_res):
        
       result = self.accuracy_strategy.append(acc_res, ignore_index = True)
       self.accuracy_strategy = result.copy()
    
    def update(self,acc, pidx):
        #infer_method = ['em', 'mv', 'mvhard','mvsoft', 'bcc']
        te = acc.groupby(['method','num_attackers']).mean()
        te = te.reset_index(level=['method', 'num_attackers'])
        for infM in self.infermethod:
            second = te[te['method'] == infM]['accuracy'].values[0]
            if infM == 'em':
                self.update_acc_perIteration[0] [pidx] += second
            elif infM == 'mv':
                self.update_acc_perIteration[1] [pidx] += second
            elif infM == 'mvsoft':
                self.update_acc_perIteration[3] [pidx] += second
            elif infM == 'mvhard':
                self.update_acc_perIteration[2] [pidx] += second
            elif infM == 'bcc':
                self.update_acc_perIteration[4] [pidx] += second
        return        


    def getcolumn_ansFile(self, path):
        with open(path, "r") as f:
            col = next(csv.reader(f))
        return col
    
    def realdata_run_partial_kn_gen_attack(self, niter, p_ad, newfile, p_ad_id, resultfile, dirname,
                                subdirectory, nworker, istargeted, target_list, knowledge_rate):   
        
        ares, success_rate= tm.recognizibility_partial(p_ad,newfile, niter, self.truthfile, dirname,
                                           subdirectory, nworker, istargeted,target_list, knowledge_rate)
        ares.to_csv(resultfile, index=False)

        

    def realdata_run_gen_attack(self, niter, p_ad, newfile, p_ad_id, resultfile, dirname,
                                subdirectory, nworker, istargeted, target_list):           
        """    
        for p_ad_id, p_ad in enumerate(self.num_adusers_percantage):
                num_adversary = round(self.numhonestUsers * 0.01 * p_ad)
                mean = 200 # round(labels / workers)
                if num_adversary > 0:
                    attacker_assignedtask = self.assignAttackerToTask(num_adversary, mean, self.ntasks)
                    adv_ans = self.poisionAttack(num_adversary,attacker_assignedtask, gamma)
                    advfile =self.datadir +'\\adv_ans_p{}_g{}_a{}_b{}.csv'.format(p_ad,gamma,self.avg_adv_alph,self.avg_adv_beta)
                    cols = self.getcolumn_ansFile(self.answerfile) 
                    df = pd.DataFrame(adv_ans, columns = cols)
                    df.to_csv(advfile, index=False)

                    newfile = self.datadir + '\\union_ans_p{}_g{}_a{}_b{}.csv'.format(p_ad,gamma,self.avg_adv_alph,self.avg_adv_beta)
                    result = pd.DataFrame(columns = cols)
                    ans = pd.read_csv(self.answerfile)
                    ans_ad = pd.read_csv(advfile)
                    result = pd.concat([ans, ans_ad], ignore_index = True)
                    result.to_csv(newfile,index = False)

                    #wholeusers_ans = self.mergeTwoCSVFile(advfile , self.answerfile, newfile)
                else:
                    newfile = self.answerfile
        """     
        ares, success_rate= tm.InferTruth_runAllMethods(p_ad,newfile, niter, self.truthfile, dirname,
                                           subdirectory, nworker, istargeted,target_list)
        ares.to_csv(resultfile, index=False)
        if istargeted:
            sfile = resultfile.with_suffix('.txt')
            success_rate.to_csv(sfile, index=False)
        #self.appendAcctodf(ares)  
        #self.update(ares, p_ad_id)
           #saveanswer
           #self.applytruthMethods(newfile)
           #accuracy
            
            
    def get_specific_Accuracy(self, infermethod, percentage_attacker_id):
        return self.update_acc_perIteration[infermethod][percentage_attacker_id]
    
    
    def readfilegetcol(self, datapath, colname):
            columns = defaultdict(list) # each value in each column is appended to a list
            with open(datapath) as f:
                reader = csv.DictReader(f) # read rows into a dictionary format
                for row in reader: # read a row as {column1: value1, column2: value2,...}
                    for (k,v) in row.items(): # go over each column name and value 
                        columns[k].append(v) # append the value into the appropriate list
                                             # based on column name k
            return list(set(columns[colname]))
