# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:21:23 2018
@author: tahma
"""
import os, random , shutil
#from collections import Counter
import generateUserTaskGraph as gutg
import generlizedpoisoning as gpas
import truth_methods as tm 
from StrategyType import strategyType
import grraphs
import pandas as pd
import numpy as np
import glob, csv
import attackonRealData  as realdbAtt
from collections import defaultdict
from pathlib import Path, PureWindowsPath
try:
    import cPickle as pickle
except:
    import pickle

np.random.seed(2830)
random.seed(6600)


class evalprogram():
    def __init__(self, whichmodel, gammas, infermethods, adv_perc ):
        
        self.labels = ['0', '1']
        self.num_labels = len(self.labels)
        self.whichModel = whichmodel # 1 realdata 0 syntahic 
        self.ResultsDir = os.path.join(os.getcwd(), 'Results')         
        
        self.iterations = 10    
        self.initialQuality = 0.7
        
        available_strategies = ['G']
        self.attack_strategy = ['G']
        self.gamma_prams = gammas
        self.available_infermethods = infermethods
        self.num_adusers_percantage = adv_perc
        self.accuracies =  [[[[0 for p in range(len(self.gamma_prams))] for p in range(len(self.num_adusers_percantage))] for a in range(len(available_strategies))] for inf in range(len(self.available_infermethods))] #inference method (em , mv)
     
        
    """
    Defines parameters and the ranges 
    taskCounts = number of tasks
    labelCounts = number of labels 
    """
    def defineParametersAndRanges(self,taskcounts, userscount,WorkerSelectionMethod, behave_gamma):
    
        # The tasks for each worker
        # The background probability vector
        #Parameters for each worker
        #The labels given by the workers
        self.usercount = userscount
        self.taskcount = taskcounts
        self.workermodel = WorkerSelectionMethod       
        #TODO: Check if I need to use beta distribution or not
        self.tasks_lable_proportion = 0.5 # I am not sure about this!!!
        self.behave_gamma = behave_gamma
        return 
    
    # IF the accuracy of MV is more than 0,5 then graph is accepted
    def checkEligibilityData(self,datafile, truthfile, threshold):
        result = tm.getAccuracyofHonestUser(datafile, truthfile)
        if result > threshold:
            return True 
        return False

    def cleanFolderContent(self, dirname):
        folder = os.path.join(os.getcwd(), dirname)                       
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)        

    def correctPath_os(self,file):
        filename = PureWindowsPath(file)
        correct_path = Path(filename)

        return correct_path
    
    def get_answerfilepath(self, whichlabel):
        if whichlabel == 1: 
            f = os.path.join(os.getcwd(), 'data\\answers.csv')
            filename = PureWindowsPath(f)
            correct_path = Path(filename)

            return correct_path
        
      
    def createusertaskgraph(self,mean_alpha, svd_alpha, mean_beta, svd_beta, graphDir, gamma, truthfile, att_adv_reliability_filepath):
        is_acc_above_thresh = False
        while is_acc_above_thresh == False:
            
            true_labels,Task_perWorker = gutg.gen_user_task_Graph(self.usercount,self.taskcount,                                                                                                                                                            
                                                                 self.tasks_lable_proportion,graphDir) 
            
            
            user_df = gutg.gen_userBehavior(self.usercount,self.taskcount,self.workermodel,
                                            self.initialQuality,mean_alpha,
                                            svd_alpha, mean_beta, svd_beta,graphDir, true_labels,Task_perWorker,
                                            mean_a_adv,mean_a_adv, gamma, att_adv_reliability_filepath)
            
            datafile = self.correctPath_os(os.path.join(os.getcwd(), 'data\\answers.csv'))
            is_acc_above_thresh = self.checkEligibilityData(datafile, truthfile , 0.5)
            if is_acc_above_thresh == False:
                self.cleanFolderContent('data')
                self.cleanFolderContent(graphDir)
            
            num_allanswers = sum([len(v) for v in Task_perWorker.values()])
            print("All labels provided by users is{}".format(num_allanswers))
            #infoFile(self.usercount,self.taskcount, num_allanswers)
            
        #graph_path = os.path.join(os.getcwd(), 'graph\\iter_0\\usertask.gml')
        return  user_df, true_labels,Task_perWorker
    
    
    def randomlychooseattacker(self, user_set,num_attacker):
        rnd_user = random.sample(user_set,num_attacker)
        user_set = list(set(user_set) - set(rnd_user))                           
        tasks_SelectedUsers = dict((k, self.Task_perWorker[k]) for k in rnd_user)
        return rnd_user, tasks_SelectedUsers
        

    """
    Parameters: 
    behavior_gamma : percentage of assigned tasks are act like the normal user 
    and 1- behavior_gamma acts like the malicious one
    adve_alpha, adv_beta : average the relibility parameters of attackers 
        # Attacker knowledge : Knowing the Normal users reliability (just the row that they are going to change)[I consider this option right now]
        # Does not know anything, so just assume that the average reliability above 0.5 and then choose one of them randomly
        # Know a partial answers of workers for some tasks
    
    """
    def run_general_poisoning_Attack(self,iteration, generalizedscenario, tid, uid, iters, 
                                     mean_a_adv,svd_a_adv,mean_b_adv, svd_b_adv,
                                     partialAnswersObj, partialAns, graphdir,mean_a_norm,
                                     mean_b_norm, gam, attackersID_list,  g_honesttasks, confusion):
        print("General poisoning attack is starting.....") 
        avg_cm = [mean_a_norm, mean_b_norm, mean_a_adv,mean_b_adv] 
        attack_df = self.user_df.copy()
        # gamma means that obfuscate greatr means that behave like normal user 
        generalizedscenario.initializeParameter( self.true_labels, self.behave_gamma )
        datafile = self.get_answerfilepath(1)
        for p_ad_id , p_ad in enumerate(self.num_adusers_percantage):            
            num_attacker = int( self.usercount * (p_ad) / 100 )
            if num_attacker > 0 :
                rnd_user = attackersID_list[0:num_attacker]                
                tasks_SelectedUsers = dict((k, self.Task_perWorker[k]) for k in rnd_user)
                remain_name = "_a{}_b{}_aprim{}_bprim{}_gamma{}_p{}.csv".format(mean_a_norm, mean_b_norm, mean_a_adv, mean_b_adv, gam, p_ad)
                savingfile_path = self.correctPath_os(os.path.join(graphdir , "answers" + remain_name))
                attack_df = generalizedscenario.CreateAttackData_increaseError(attack_df,rnd_user,
                                                                               tasks_SelectedUsers,datafile, self.usercount, p_ad_id, iters,
                                                                               mean_a_adv,svd_a_adv,mean_b_adv,svd_b_adv, savingfile_path,
                                                                               g_honesttasks, confusion ) 
                #prev_rnd_user = rnd_user
                #prec_selectedTasks = tasks_SelectedUsers
                          
            #old_prcent_adversary = p_ad
            truthfile = self.correctPath_os(os.path.join(os.getcwd(), 'data\\truth.csv'))
            ares  = tm.applyTruthInferenceMethods(p_ad,datafile, iteration,
                                                                 self.true_labels, truthfile,
                                                                 graphdir, avg_cm,gam)
            generalizedscenario.appendAcctodf(ares)  
            #generalizedscenario.update(ares, p_ad_id)
            #0 em , 1 mv
        print("General poisoning attack is done for iteration {} !".format(iteration))


    def getAverageTaskPerUsers(self, userassignedtask):
        avg = 0.0
        for k,v in userassignedtask.items():
            avg += v            
        return avg/ len(userassignedtask)

    def initializeParameterforAllStrategy(self):
        default_num_adve = 0
        attackerDataframe = self.user_df.copy()
        return default_num_adve, attackerDataframe
    

        
        
        
    def writeResults(self, resultname, model_accuracies):
        model_accuracies.to_csv(resultname, index=False)

    
    
    #exp_params = "advAl{}_advB{}_normAl{}_normB{}_gamma{}_landa{}_pw{}_pt{}_u{}_t{}".format(avg_alpha_adversary, avg_beta_adversary, avg_alpha_normaluser, avg_beta_normaluser, behave_gamma,landa,partialtasks, partialworkers, uidx, tidx)
    def printresults(dirfiles, avg_alpha_adv, avg_beta_adv, avg_alpha_norm, avg_beta_norm, gamma,
            landa,partialtasks, partialworkers, perc_adv_id, nuser, ntasks, variedfeature,variedIndex):            

            method_nums = 5
            exp_params =  "advAl{}_advB{}_normAl{}_normB{}_gamma{}_landa{}_pw{}_pt{}_u{}_t{}".format(avg_alpha_adv, avg_beta_adv, avg_alpha_norm, avg_beta_norm, gamma,landa,partialtasks, partialworkers, nuser, ntasks)
           
                    
            title = "The affect of with fix advAl{}_advB{}_normAl{}_normB{}_gamma{}_landa{}_pw{}_pt{}_u{}_t{}".format(avg_alpha_adv, avg_beta_adv, avg_alpha_norm, avg_beta_norm, gamma,landa,partialtasks, partialworkers, nuser, ntasks)
            
            
            xaxies = []
            yaxies = [[] for i in range(method_nums)]
            
    #advAl0.2_advB0.1_normAl0.8_normB0.7_gamma1_landa1_pw1.0_pt0.9_u0_t0pidx_4
            for name in glob.glob(dirfiles):
                for value in variedIndex:
                    if variedfeature == 'gamma':
                        exp_params =  "advAl{}_advB{}_normAl{}_normB{}_gamma{}_landa{}_pw{}_pt{}_u{}_t{}".format(avg_alpha_adv, avg_beta_adv, avg_alpha_norm, avg_beta_norm, value , landa , partialtasks, partialworkers, nuser, ntasks)
                        pattern = exp_params + 'pidx_'+ str(perc_adv_id) +'.csv'  
                        if name.split('\\')[-1] == pattern:
                            df = pd.read_csv(name)
                            xaxies.append(value)
                            #MV
                            tmpdf = df['accuracy'][(df['num_attackers']== perc_adv_id) & (df['method']== 'mv')].values[0]
                            tmpdf = '%.3f' % tmpdf
                            yaxies[0].append(float(tmpdf))
                            # EM
                            tmpdf = df['accuracy'][(df['num_attackers']== perc_adv_id) & (df['method']== 'em')].values[0]
                            tmpdf = '%.3f' % tmpdf
                            yaxies[1].append(float(tmpdf))
                        
            grraphs.plotTwoList(xaxies, yaxies[0], yaxies[1], variedIndex, title)
            return 

    
    # write to file with cPickle/pickle (as binary)
    def ld_writeDicts(self,filePath,mdict):
        f=open(filePath,'wb')
        newData = pickle.dumps(mdict, 1)
        f.write(newData)
        f.close()
    
    
    
    #############
    
    # read file decoding with cPickle/pickle (as binary)
    def ld_readDicts(self,filePath):
        f=open(filePath,'rb')
        data = pickle.load(f)
        f.close()
        return data
    
    # return dict data to new dict
    #newDataDict = ld_readDicts('C:/Users/Lee/Desktop/test2.dta')
    
    def assignHonestTasksPerGamma(self,uids, gammalist, user_tasks, path):
        dg = {}
        for u in uids: 
            dg[u] = {}
            for g in gammalist: 
                sample_size = round(g * len(user_tasks[u]))
                if sample_size < len(user_tasks[u]):
                    dg[u][g] = user_tasks[u][0:sample_size]
                else: 
                    dg[u][g] = user_tasks[u]
        self.ld_writeDicts(path,dg)
        return dg

        
    
    

    
    def runevaluation(self, u_idx, t_idx,mean_a_norm,svd_a_norm,  mean_b_norm, svd_b_norm,
                      mean_a_adv,svd_a_adv, mean_b_adv, svd_b_adv, exp_param, niter, gam, gamma_list):
        

        # Generalized Method
        generalize_attack = gpas.AttackStrategy()
        datadir = os.path.join(os.getcwd(), 'data')            

        # Create the worker-task assignment graph
        gf = 'graph\\iter_{}_uid_{}_tid_{}'.format(niter, u_idx, t_idx)
        graph_dir = os.path.join(os.getcwd(), gf)

        truthfilepath = self.correctPath_os(self.correctPath_os(graph_dir+"\\truth.csv"))

        remain_name = '_a{}_b{}.csv'.format(mean_a_norm, mean_b_norm)
        
        att_adv_reliability_filepath = self.correctPath_os(graph_dir+'\\rel_adv_a{}_adv_b{}.pkl'.format(mean_a_adv, mean_b_adv))

        if os.path.exists(self.correctPath_os(graph_dir+'\\taskperworkerDict.npy')) == False:
            self.user_df, self.true_labels,self.Task_perWorker = self.createusertaskgraph(mean_a_norm,svd_a_norm,mean_b_norm,svd_b_norm,graph_dir,gam, truthfilepath, att_adv_reliability_filepath)
            attackersID_list = [i for i in range(u_idx)]
            random.shuffle(attackersID_list)
            
            
            glists_path = self.correctPath_os(graph_dir+'\\honestTasks.dta')
            g_honesttasks = self.assignHonestTasksPerGamma(attackersID_list, gamma_list, self.Task_perWorker,  glists_path)
            
                        
            num_allanswers = sum([len(v) for v in self.Task_perWorker.values()])
            rem_name = '_a{}_b{}__aprim{}_bprim{}_gamma{}_p{}.txt'.format(mean_a_norm, mean_b_norm,mean_a_adv, mean_b_adv, self.behave_gamma,0)
            infpath = self.correctPath_os(graph_dir + '\\statInfoFile'+ rem_name)
            with open(infpath, 'a') as statInfoFile:
                statInfoFile.write("All labels provided by users is{} and nuser is {} ntasks {} iteration {}\n".format(num_allanswers, u_idx, t_idx, niter))     
            
            infpath = self.correctPath_os(graph_dir + '\\attackerList.txt')
            with open(infpath, 'w') as filehandle:  
                for listitem in attackersID_list:
                    filehandle.write('%s\n' % listitem)    
        else:
            print("set the user behavior")
            
            tdata = pd.read_csv(self.correctPath_os(graph_dir+"\\truth.csv"), names='e')
            tmptruth = list(tdata['e']) 
            tmptruth = tmptruth[1:]
            tmptruth = [float(i) for i in tmptruth]
            self.true_labels = tmptruth
                
            self.Task_perWorker = np.load(self.correctPath_os(graph_dir+'\\taskperworkerDict.npy')).item()
            
            glists_path = self.correctPath_os(graph_dir+'\\honestTasks.dta')
            g_honesttasks = self.ld_readDicts(glists_path)
            
            attackersID_list = []
            att_list_path = self.correctPath_os(graph_dir + '\\attackerList.txt')
            with open(att_list_path, 'r') as filehandle:  
                for line in filehandle:
                    currentPlace = line[:-1]            
                    attackersID_list.append(int(currentPlace))
            
            
            reliabilityfile = '\\answers_a{}_b{}.csv'.format(mean_a_norm, mean_b_norm)
            if os.path.exists(self.correctPath_os(graph_dir+reliabilityfile)) == False:
                gutg.gen_userBehavior(self.usercount,self.taskcount,self.workermodel,
                                      self.initialQuality,mean_a_norm,svd_a_norm,mean_b_norm,
                                      svd_b_norm,graph_dir,self.true_labels,self.Task_perWorker,
                                      mean_a_adv,mean_b_adv, self.behave_gamma, att_adv_reliability_filepath)
            else:
                gutg.load_userBehavior(mean_a_norm,mean_b_norm,graph_dir,
                                       mean_a_adv,mean_b_adv, self.behave_gamma)
        
        if os.path.exists(correctPath_os(att_adv_reliability_filepath)) == False:
            adv_conf_matrix_list = gutg.adv_model_AssignReliability(attackersID_list,  mean_a_adv, svd_a_adv, mean_b_adv, svd_b_adv)
            with open(att_adv_reliability_filepath, 'wb') as output:
                pickle.dump(adv_conf_matrix_list, output, pickle.HIGHEST_PROTOCOL)
        
        with open(att_adv_reliability_filepath, 'rb') as input:
                confusion = pickle.load(input)
             
        shutil.copy(self.correctPath_os(graph_dir+"\\truth.csv"), self.correctPath_os(datadir+"\\truth.csv"))  
                   
        shutil.copy(self.correctPath_os(graph_dir+"\\answers"+remain_name), self.correctPath_os(datadir+"\\answers.csv"))
        self.user_df = pd.read_csv(graph_dir+"\\user_df"+remain_name) 	
        partialAnswers = []
        length_dict = {key: len(value) for key, value in self.Task_perWorker.items()}
        print("average task per user {}\n".format(self.getAverageTaskPerUsers(length_dict))) 
        for t in self.attack_strategy:
            	if t == 'G':
                    self.run_general_poisoning_Attack(niter, generalize_attack, u_idx, t_idx,
                                                      self.iterations,mean_a_adv,svd_a_adv,
                                                      mean_b_adv, svd_b_adv, partialAnswers, 0,
                                                      graph_dir, mean_a_norm, mean_b_norm,
                                                      self.behave_gamma, attackersID_list, g_honesttasks, confusion)
            		
                    
    
        
        #infermethodsdic = {0:'em', 1: 'mv', 2:'mvhard', 3:'mvsoft', 4:'bcc', 5:'lfc', 6:'kos', 7:'zc'}

        """
        for infer, v in infermethodsdic.items(): 
            for pidx in range(len(self.num_adusers_percantage)):               
                self.accuracies[infer][strategyType.general.value][pidx]= generalize_attack.get_specific_Accuracy(infer,pidx)
        """
        
        #for pidx in range(len(self.num_adusers_percantage)):
        resultname = self.correctPath_os(self.ResultsDir + '//'+ exp_param + 'gamma_'+ str( self.behave_gamma) +'.csv' )     
        self.writeResults(resultname, generalize_attack.getAccuracy())
        accpath =self.correctPath_os( graph_dir + '\\' + exp_param + '_allaccuracies.csv')        
        generalize_attack.saveAllAccuracy(accpath)
        
                            

def getcolumn_ansFile(path):
    with open(path, "r") as f:
         col = next(csv.reader(f))
    return col

def readfilegetcol(datapath, colname):
    columns = defaultdict(list) # each value in each column is appended to a list
    with open(datapath) as f:
          reader = csv.DictReader(f) # read rows into a dictionary format
          for row in reader: # read a row as {column1: value1, column2: value2,...}
              for (k,v) in row.items(): # go over each column name and value 
                   columns[k].append(v) # append the value into the appropriate list
                                             # based on column name k
    return list(set(columns[colname]))

def correctPath_os(file):
    filename = PureWindowsPath(file)
    correct_path = Path(filename)

    return correct_path
          
def gete2wlandw2el(datafile):
    e2wl = {}
    w2el = {}
    label_set=[]
    
    f = open(datafile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        example, worker, label = line
        if example not in e2wl:
            e2wl[example] = []
        e2wl[example].append([worker,label])

        if worker not in w2el:
            w2el[worker] = []
        w2el[worker].append([example,label])

        if label not in label_set:
            label_set.append(label)

    return e2wl,w2el,label_set

def prepare_partial_files(rate_knowledge, ansfile):

    e2wl,w2el,label_set = gete2wlandw2el(ansfile)    
    users_dict ={}

    k_list = list(w2el.keys())
    for i, u  in enumerate(k_list):
        users_dict[i] = k_list[i]

    all_users = len(k_list)
    
    num_knowledge = int(all_users * rate_knowledge)
    
    range_users = [i for i in range(all_users)]
            
    kn_range = list(set(range_users))
    
    selected_kn = np.random.choice(kn_range, num_knowledge, replace = False) 
    
    kn_dict = {}
    for uid in selected_kn: 
        kn_dict[users_dict[uid]] = w2el[users_dict[uid]]
        
    cols = getcolumn_ansFile(ansfile)
    seprated_kn= pd.DataFrame(columns = cols)
    
    for u in kn_dict:
        for t in kn_dict[u]:
            row = [t[0], u ,t[1]]  
            seprated_kn.loc[len(seprated_kn)] = row
 
     
    return seprated_kn
    
if __name__ == '__main__': 

    taskuserinfo={}

    targeted = True
    
    
    partial_recognizability = False
    knowledge_rate = 1.0
    
    """
    For the modified we need: 
    
    For the injected, we should first run ...
    """
    type_targeted = 'm' # 'j' injected or 'm' modified

    svd_a_norm = 0.002
    svd_a_adv = 0.002
    landa_prams = [0]
    infer_method = ['mv','em','mvsoft','mvhard', 'bcc', 'lfc', 'kos', 'zc']
       
    dataSet_type = 1 # real == 1 or syntethic == 0
    
    
    #datasetName = "product" #d_sentiment
    datasetName = "d_sentiment"    
    
    partial_kn = False
    
    
    knowledge = [1.0]
    whichdataset = 0 #product == 1, d == 0
    if dataSet_type == 1:
        behave_gamma_prams = [0]#0.7, 0.8, 0.9, 1.0]#,0.1, 0.2, 0.3,0.4, 0.5, 0.6 , 0.7]#,0.04, 0.06, 0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # [0, 0.2, 0.4, 0.6, 0.8,1]    0 % honest 100 adverse or 20% honest 

        if type_targeted == 'm': 
            attackerPercentage = [ 30]#40,50, 60]#[0, 10, 20, 30, 40,50, 60]
        else: 
            if partial_kn :
                attackerPercentage = [20.0]
                knowledge = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 , 0.7, 0.8, 0.9, 1] #[0, 0.2, 0.3, 0.4, 0.5] # [0, 0.2, 0.4, 0.6, 0.8,1]    0 % honest 100 adverse or 20% honest 
                behave_gamma_prams =[0, 0.1, 0.2]
            else:
                attackerPercentage = [30]
                #attackerPercentage =  [49,52,53,54,55]
#[0, 10, 20, 30, 40,50, 60, 70,100,140]


        if partial_recognizability : 
            attackerPercentage =[10]
            
    else: 
        attackerPercentage = [0, 3, 5, 10, 20, 30, 40,50, 60, 70, 100]
        #behave_gamma_prams = [0,0.04, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 , 0.7] #[0, 0.2, 0.3, 0.4, 0.5] # [0, 0.2, 0.4, 0.6, 0.8,1]    0 % honest 100 adverse or 20% honest 
        behave_gamma_prams = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6 , 0.7] #[0, 0.2, 0.3, 0.4, 0.5] # [0, 0.2, 0.4, 0.6, 0.8,1]    0 % honest 100 adverse or 20% honest 
    
    evaluation = evalprogram(whichdataset, behave_gamma_prams, infer_method, attackerPercentage)
    iterations = 10

    
    ############################################################# Real Dataset - Injection #################################################
    if dataSet_type == 1:
        if datasetName == "product":
            #normbehave = [(1.0, 1.0), (0.95, 0.75), (0.85, 0.65), (0.9, 0.6), (0.75, 0.75)] #[1.0, 0.8, 0.6] [0.9,0.9]
            #advbehave = [(0.0,0.0),  (0.2,0.1), (0.2, 0.0), (0.3, 0.0), (0.5, 0.0)] 
            
            normbehave = [(1.0, 1.0)]
            advbehave = [(0.0,0.0)]#, (1.0,0.0), (0.0,1.0),(0.0,0.9),(0.8,0.0),(0.9,0.0)] 
            """
            normbehave = [(1.0, 1.0)]#, (0.85, 0.70)]
            advbehave = [(0.2,0.1), (0.2,0.0)] 
            #advbehave = [(0.2,0.1), (0.2,0.0)] 
            """


        else:
            normbehave = [(1.0, 1.0)]#, (0.9, 0.9), (0.85, 0.85), (0.8, 0.8), (0.75, 0.75), (0.7, 0.7)] #[1.0, 0.8, 0.6] [0.9,0.9]
            advbehave = [(0.0,0.0)]#, (0.1,0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)] 
            """
            normbehave = [(1.0, 1.0), (0.9, 0.9), (0.85, 0.85), (0.8, 0.8), (0.75, 0.75), (0.7, 0.7)] #[1.0, 0.8, 0.6] [0.9,0.9]
            advbehave = [(0.0,0.0),  (0.1,0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)] 
            """
        
        datadir = correctPath_os(os.path.join(os.getcwd(), 'realdataset' , datasetName))
        truthfile = correctPath_os(os.path.join(datadir , 'truth.csv'))
        print("truth ", truthfile)
        ansfile =  correctPath_os(os.path.join(datadir ,'answers.csv'))
        if datasetName == "product":
            tasksList = readfilegetcol(truthfile, 'question')
        elif datasetName == "d_sentiment":
             tasksList = readfilegetcol(truthfile, 'q')
        honest_users_List = readfilegetcol(ansfile,'worker')
        ntasks = len(tasksList)
        infer_method = ['mv','em','mvsoft','mvhard', 'bcc', 'lfc', 'kos', 'zc']


        
        subtasks = []
        if datasetName == 'product':
            subtasks = [0.05, 0.1]#, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6] #[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,0.45, 0.5]#[0.0025, 0.64, 0.7]#0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
        elif datasetName == 'd_sentiment':
            subtasks = [0.99]#, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6] #[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,0.45, 0.5]#[0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]

        
        target_list = []
        cols = getcolumn_ansFile(ansfile)
        for kn in knowledge: 
            for niter in range(iterations):
                print("_________________________ITeration_{}____ is starting".format(niter))
                for gidx, gamma in enumerate(behave_gamma_prams):
                    for norm in normbehave:
                        for adv in advbehave:  
                            #######################################Targeted Section############################################
                            ##############################################################################################
                            ##############################################################################################
                            
                            if targeted == True:
                                print("Targeted Mode")
                                for st in subtasks:
                                    print("_________________________subtask_{}____ is starting".format(st))
                                    target_list = []
                                    dirname = os.path.join(datadir , "targeted_iteration_{}".format(niter), 'iter_{}_{}'.format(niter, st))
                                    
                                    tarfile = correctPath_os(os.path.join(dirname ,'targetLists.txt'))
                                    with open(tarfile, "r") as f:
                                          for line in f:
                                            target_list.append(line.strip())
    
                                    filename = correctPath_os(os.path.join(dirname , "g_{}_alpha_{}_beta{}_adalpha{}_adbeta{}.csv".format(gamma, norm[0], norm[1], adv[0], adv[1])))
                                    for pid, p in enumerate(attackerPercentage):
                                        nrows = 0
                                        n_adv = int(p* 0.01 * len(honest_users_List))
                                        group_names = []
                                        adv_ans = []
#                                        if p > 0: 
#                                            df = pd.read_csv(filename)
#                                            gb = df.groupby(['worker']).count()
#                                            df_list = list(df.groupby(['worker']))
#                                            for i in range(n_adv):
#                                                if type_targeted == 'm': 
#                                                    id_adv = gb.index[i]
#                                                    group_names.append(id_adv)
#                                                    if len(adv_ans) > 0: 
#                                                        adv_ans = pd.concat( [adv_ans, df_list[i][1]], ignore_index= True)
#                                                    else:
#                                                        adv_ans = df_list[i][1]
#                                                else: 
#                                                    id_adv = 'adv_{}'.format(i)
#                                                    nrows += gb['answer'][id_adv]
#                                                    adv_ans = df[:nrows]
                                        subdirectory = "g_{}_alpha_{}_beta{}_adalpha{}_adbeta{}".format(gamma, norm[0],
                                                          norm[1], adv[0], adv[1])
                                        if not os.path.exists( correctPath_os(os.path.join(dirname +'\\'+ subdirectory))):
                                            os.makedirs( correctPath_os(os.path.join(dirname +'\\'+ subdirectory)))
                                        infpath = correctPath_os(os.path.join(dirname , subdirectory, 'statInfoFile_p{}.txt'))
                                        newfile = correctPath_os(os.path.join(dirname , subdirectory ,'union_ans_p{}.csv'.format(p)))
                                        accresultfile = correctPath_os(os.path.join(dirname , subdirectory ,'accresult_ans_p{}.csv'.format(p)))
                                        result = pd.DataFrame(columns = cols)
                                        ans = pd.read_csv(ansfile)
                                        if p > 0:
                                            if type_targeted == 'm':
                                                grouped = ans.groupby(['worker'])
                                                for group_name in group_names:
                                                    ans = ans.drop(grouped.get_group(group_name).index)
                                                result = pd.concat([ans, adv_ans], ignore_index = True)
                                            else: 
                                                result = pd.concat([ans, adv_ans], ignore_index = True)
                                        else:
                                            result = ans
                                        result.to_csv(newfile,index = False)
                                        nworkers = n_adv+ len(honest_users_List) 
                                        attackScenario = realdbAtt.AttackStrategy(attackerPercentage,infer_method,truthfile)
                                        attackScenario.realdata_run_gen_attack(niter, p, newfile, pid, accresultfile,
                                                                               dirname , subdirectory, nworkers, targeted, target_list)
                            ####################################### UnTargeted Section ############################################
                            ##############################################################################################
                            ##############################################################################################
                            
                            else: 
                                #dirname = os.path.join(datadir , "iter_{}".format(niter))
                                if partial_kn :
                                    dirname = os.path.join(datadir , 'inj_inf_iter_{}'.format(niter), 'partially_inj_untargeted_kn_{}'.format(kn))
                                else:  
                                    dirname = os.path.join(datadir , "untargeted_inj_iteration_{}".format(niter))
                                    #dirname = os.path.join(datadir ,'Untargeted_injected',  "iter_{}_untar_inj_senti".format(niter))
                                
                                filename = correctPath_os(os.path.join(dirname , "g_{}_alpha_{}_beta{}_adalpha{}_adbeta{}".format(gamma, norm[0], norm[1], adv[0], adv[1])))
                                for pid, p in enumerate(attackerPercentage):
                                    nrows = 0
                                    n_adv = int(p * 0.01 * len(honest_users_List))
                                    group_names = []
                                    adv_ans = []
                                    if p > 0: 
    #                                    df = pd.read_csv(filename)
    #                                    gb = df.groupby(['worker']).count()
    #                                    df_list = list(df.groupby(['worker']))
    #                                    for i in range(n_adv):
    #                                        ####################################### Modified ############################################
    #                                        ##############################################################################################
    #                                        ##############################################################################################
    #                                        if type_targeted == 'm': 
    #                                            id_adv = gb.index[i]
    #                                            group_names.append(id_adv)
    #                                            if len(adv_ans) > 0: 
    #                                                adv_ans = pd.concat( [adv_ans, df_list[i][1]], ignore_index= True)
    #                                            else:
    #                                                adv_ans = df_list[i][1]
    #                                        ####################################### Injected ############################################
    #                                        ##############################################################################################
    #                                        ##############################################################################################
    #                                        
    #                                    else: 
    #                                            id_adv = 'adv_{}'.format(i)
    #                                            nrows += gb['answer'][id_adv]
    #                                            adv_ans = df[:nrows]
                                            
                                            adv_subdirectory = "g_{}_alpha_{}_beta{}_adalpha{}_adbeta{}_adv{}.csv".format(gamma, norm[0],
                                                  norm[1], adv[0], adv[1], p)
                                
                                            adv_created_path = os.path.join(dirname  , adv_subdirectory)
                                            
                                            adv_ans = pd.read_csv(adv_created_path)
                                            
                                    subdirectory = "g_{}_alpha_{}_beta{}_adalpha{}_adbeta{}".format(gamma, norm[0],
                                                      norm[1], adv[0], adv[1])
                                    if not os.path.exists( correctPath_os(os.path.join(dirname , subdirectory))):
                                        os.makedirs( correctPath_os(os.path.join(dirname ,subdirectory)))
                                    infpath = correctPath_os(os.path.join(dirname , subdirectory, 'statInfoFile_p{}.txt'))
                                    newfile = correctPath_os(os.path.join(dirname , subdirectory ,'union_ans_p{}.csv'.format(p)))
                                    accresultfile = correctPath_os(os.path.join(dirname , subdirectory ,'new_accresult_ans_p{}.csv'.format(p)))
                                    result = pd.DataFrame(columns = cols)
#                                    if partial_kn == True: 
#                                        ans = prepare_partial_files(knowledge_rate, ansfile)
#                                        newfile = correctPath_os(os.path.join(dirname , subdirectory ,'kn_union_ans_p{}_kn{}.csv'.format(p, knowledge_rate)))
#                                        accresultfile = correctPath_os(os.path.join(dirname , subdirectory ,'kn_accresult_ans_p{}_kn{}.csv'.format(p, knowledge_rate)))
#                                    
#                                    else:
                                    
                                    ans = pd.read_csv(ansfile)
                                    if p > 0:
                                        
                                        if type_targeted == 'm':
                                            grouped = ans.groupby(['worker'])
                                            for group_name in group_names:
                                                ans = ans.drop(grouped.get_group(group_name).index)
                                            result = pd.concat([ans, adv_ans], ignore_index = True)
                                        else: 
                                            if partial_kn == False: 
                                               result = pd.concat([ans, adv_ans], ignore_index = True)
                                    else:
                                        result = ans
                                    
                                    
                                    if partial_kn == False: 
                                        result.to_csv(newfile,index = False)
                                    else:
                                        shutil.copy(adv_created_path, newfile)  
                                    

                                        
                                    nworkers = n_adv + int(len(honest_users_List) * knowledge_rate)
                                    attackScenario = realdbAtt.AttackStrategy(attackerPercentage,infer_method,truthfile)
                                    #if partial_recognizability == False:
                                    attackScenario.realdata_run_gen_attack(niter, p, newfile, pid, accresultfile,
                                                                           dirname , subdirectory, nworkers, targeted, target_list)
                                    #else:
                                    #    attackScenario.realdata_run_partial_kn_gen_attack(niter, p, newfile, pid, accresultfile,
                                    #                                       dirname , subdirectory, nworkers, targeted, target_list, knowledge_rate)
    ############################################################# Syn Dataset #################################################
    
    else:
        taskcounts = 400
        userscount = 20
        density = [20, 15, 10, 5, 3]
		
        normbehave = [1.0] #,0.95, 0.85, 0.75, 0.65]
        advbehave = [0.0] #, 0.1, 0.2, 0.3, 0.4]
        for niter in range(iterations):
            for behave_gamma in behave_gamma_prams:
                for mean_a_norm in normbehave:
                    for mean_a_adv in advbehave:
                        WorkerSelectionMethod = 'CM'# sys.argv[3]
                        exp_params = "advAl{}_advB{}_normAl{}_normB{}_gamma{}_u{}_t{}".format(mean_a_adv, mean_a_adv, mean_a_norm, mean_a_norm, behave_gamma,userscount, taskcounts)
                        evaluation.defineParametersAndRanges(taskcounts, userscount,WorkerSelectionMethod, behave_gamma)                        
                        evaluation.runevaluation(userscount, taskcounts, mean_a_norm,svd_a_norm, mean_a_norm, svd_a_norm,mean_a_adv,svd_a_adv, mean_a_adv, svd_a_adv, exp_params, niter, behave_gamma, behave_gamma_prams)
        							
               
    


    
    
