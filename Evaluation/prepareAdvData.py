# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:17:12 2018

@author: tahma
"""

import csv, os, random, math, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from confusion_matrix import  Confusion_Matrix  as cm

import seaborn as sns
from scipy import stats
import matplotlib.pylab as plt



def assign_adversary_reliability(numAttacker, avg_adv_alph,avg_adv_beta ):
        adversaryConfusioMatrix = cm()
        #default : 0.002
        svd_alpha = 0.002
        # change the svd[0, 0.01, 0.1]
        svd_beta = 0.002
        adversaryConfusioMatrix.createCM(numAttacker, avg_adv_alph[0],svd_alpha,avg_adv_beta[1], svd_beta)        
        return adversaryConfusioMatrix
    
def assign_Normal_reliability(numAttacker, norm_alph,norm_beta ):
        adversaryConfusioMatrix = cm()
        adversaryConfusioMatrix.createCM(numAttacker, norm_alph[0],0.002,norm_beta[1], 0.002)
        
        return adversaryConfusioMatrix




def getlabelbasedCM(alpha, beta, truth):  
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
 

def assignLabel(adv, adversayReliability_alpha, adversayReliability_beta, assignedTasks, normBehave_tidx, advBehave_tidx, norm_alpha, norm_beta,  tasks_dict, which ):
               
            line = []
            if len(advBehave_tidx) != 0:
                for ctask in advBehave_tidx:
                         label = getlabelbasedCM(adversayReliability_alpha, adversayReliability_beta ,tasks_dict[ctask]['truth'] )
                         if which == 'product':
                             task_name = tasks_dict[ctask]['question']
                         elif which =='d_sentiment':
                             task_name = tasks_dict[ctask]['q']
                         line.append([task_name ,adv,  label])
            
            if len(normBehave_tidx) != 0:
                for ctask in normBehave_tidx:
                    #print("Attackers behave like normal users")               
                    label = getlabelbasedCM(norm_alpha, norm_beta ,tasks_dict[ctask]['truth'] )
                    if which == 'product':
                                 task_name = tasks_dict[ctask]['question']
                    elif which =='d_sentiment':
                                 task_name = tasks_dict[ctask]['q']
                    line.append([task_name, adv, label])
                """
                with open("output.csv", "w", newline="") as f:
                    writer = csv.writer(f)q1
                    writer.writerows(line)
                """
            return line
        
def getindextask(list1, percentage):
        n = int(len(list1) * percentage)
        partial = random.sample(list1,n)
        return partial


def poisionAttack(num_adversary, assignedTask, obfuscate,gid, truth, tasksdict, which, all_normBehave_tidx,
                  norm_adversay_behave, adversayReliability):
            advAnswers = []
            #adversayReliability = assign_adversary_reliability(num_adversary, adv_avg_cm[0], adv_avg_cm[1])
            #norm_adversay_behave = assign_Normal_reliability(num_adversary, norm_avg_cm[0], norm_avg_cm[1])
            
            
            for i in range(num_adversary):
                idx = 'adv_{}'.format(i)
                #print(idx)
                #print(adversayReliability.TTcdf[i])
                if len(all_normBehave_tidx)>0:
                    normBehave_tidx = all_normBehave_tidx[idx][gid]
                else:
                    normBehave_tidx = set()
                advBehave_tidx = set(assignedTask[idx]).difference(normBehave_tidx)
                
                advAnswers += assignLabel(idx, adversayReliability.TTcdf[i],
                                               adversayReliability.FFcdf[i], assignedTask[idx],
                                               normBehave_tidx, advBehave_tidx,
                                               norm_adversay_behave.TTcdf[i],
                                               norm_adversay_behave.FFcdf[i], tasksdict, which)
                
            return advAnswers


"""old_version
def assigntasksperusers(num_adversary, mean, ntasks, numhonestUsers):
            #assign_task name to id
            #xx = rndm(1, 1000, g=0.25, size=int(140*0.85)) 
            shape, scale = getMultiplyNums(mean)
            assigned = np.random.gamma(shape, scale, (num_adversary + numhonestUsers))
            num_assigned = random.sample(list(assigned), num_adversary)
            taskperadversary = {}
            for i, m in enumerate(num_assigned):
                tasksID = random.sample(range(0, ntasks), int(m))
                adversary_id = 'adv_{}'.format(i)
                taskperadversary[adversary_id] = tasksID            
            return taskperadversary
"""


def getDistributionNormalData(normal_data, dataset):
    if dataset == 'd_sentiment':
        
        normal_data = os.path.join(os.getcwd(), 'realdataset', 'd_sentiment', 'answers.csv')
        df = pd.read_csv(normal_data)       
        
        df.hist(column='answer', bins = 200)
        freq = df.groupby('worker').count()
        freq = freq.sort_values(by=['id'])
        max_freq = max(freq['id'])
        min_freq = min(freq['id'])
        
        freq.hist(column = 'id', bins = (max_freq-min_freq)//75)
        
    
        num_bins = (max_freq-min_freq)//75
        # plot normed histogram
        #plt.hist(freq['question'], normed=True, bins= num_bins)
    
        # find minimum and maximum of xticks, so we know
        # where we should compute theoretical distribution
        xt = plt.xticks()[0]  
        xmin, xmax = min(xt), max(xt)  
        lnspc = np.linspace(xmin, xmax, len(freq['id']))
    
        # lets try the normal distribution first
    #    m, s = stats.norm.fit(freq['question']) # get mean and standard deviation  
    #    pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
    #    plt.plot(lnspc, pdf_g, label="Norm") # plot it
    
    #    # exactly same as above
        ag,bg,cg = stats.gamma.fit(freq['id'])  
        pdf_gamma = stats.gamma.pdf(lnspc, ag, bg,cg)  
#        plt.plot(lnspc, pdf_gamma, label="Gamma")
         # exactly same as above
#        ag,bg,cg = stats.powerlaw.fit(freq['id'])  
#        pdf_powerlaw = stats.powerlaw.pdf(lnspc, ag, bg,cg)  
#        plt.plot(lnspc, pdf_powerlaw, label="POWERLAW")
#        plt.show()

    
    #    # guess what :) 
#        ab,bb,cb,db = stats.beta.fit(freq['id'])  
#        pdf_beta = stats.beta.pdf(lnspc, ab, bb,cb, db)  
#        plt.plot(lnspc, pdf_beta, label="Beta")
#    
        
      #    sns.distplot(freq['question'], fit = gamma, kde=False, 
    #             bins=(max_freq-min_freq)//50, color = 'darkblue', 
    #             hist_kws={'edgecolor':'black'},
    #             kde_kws={'linewidth': 4})
    #    
    #    
    #    params = gamma.fit(freq['question'])
    #    params = gamma.fit(freq['question'], loc=0)
    #    
#        sns.distplot(freq['id'], fit = stats.beta, hist=True, kde=False, 
#                 bins=(max_freq-min_freq)//75, color = 'darkblue', 
#                 hist_kws={'edgecolor':'black'},
#                 kde_kws={'linewidth': 5})
        #return ab,bb,cb,db 
        return ag,bg,cg 

    
    elif dataset == 'product':
            normal_data = os.path.join(os.getcwd(), 'realdataset', dataset, 'answers.csv')
            df = pd.read_csv(normal_data)
            #ax = df.plot.hist(bins=12, alpha=0.5)
            
            
            df.hist(column='answer', bins = 200)
            freq = df.groupby('worker').count()
            freq = freq.sort_values(by=['question'])
            max_freq = max(freq['question'])
            min_freq = min(freq['question'])
            
            freq.hist(column = 'question', bins = (max_freq-min_freq)//75)
            
        
            num_bins = (max_freq-min_freq)//75
            # plot normed histogram
            #plt.hist(freq['question'], normed=True, bins= num_bins)
        
            # find minimum and maximum of xticks, so we know
            # where we should compute theoretical distribution
            xt = plt.xticks()[0]  
            xmin, xmax = min(xt), max(xt)  
            lnspc = np.linspace(xmin, xmax, len(freq['question']))
        
            # lets try the normal distribution first
        #    m, s = stats.norm.fit(freq['question']) # get mean and standard deviation  
        #    pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
        #    plt.plot(lnspc, pdf_g, label="Norm") # plot it
        
        #    # exactly same as above
            ag,bg,cg = stats.gamma.fit(freq['question'])  
            pdf_gamma = stats.gamma.pdf(lnspc, ag, bg,cg)  
            #plt.plot(lnspc, pdf_gamma, label="Gamma")
            
             # exactly same as above
        #    ag,bg,cg = stats.powerlaw.fit(freq['question'])  
        #    pdf_powerlaw = stats.powerlaw.pdf(lnspc, ag, bg,cg)  
        #    plt.plot(lnspc, pdf_powerlaw, label="POWERLAW")
        
        #    # guess what :) 
        #    ab,bb,cb,db = stats.beta.fit(freq['question'])  
        #    pdf_beta = stats.beta.pdf(lnspc, ab, bb,cb, db)  
        #    plt.plot(lnspc, pdf_beta, label="Beta")
        
            #plt.show()  
            
          #    sns.distplot(freq['question'], fit = gamma, kde=False, 
        #             bins=(max_freq-min_freq)//50, color = 'darkblue', 
        #             hist_kws={'edgecolor':'black'},
        #             kde_kws={'linewidth': 4})
        #    
        #    
        #    params = gamma.fit(freq['question'])
        #    params = gamma.fit(freq['question'], loc=0)
        #    
        #    sns.distplot(freq['question'], hist=True, kde=True, 
        #             bins=(max_freq-min_freq)//70, color = 'darkblue', 
        #             hist_kws={'edgecolor':'black'},
        #             kde_kws={'linewidth': 5})
            return ag,bg,cg

def gen_graph_mal_workers(dataset, num_mal_workers, avg_labels, ansfile): 
    
    if dataset == 'product': 
        #number of tasks per mal_workers
        ag,bg,cg = getDistributionNormalData(ansfile, dataset)
    
        samples = stats.gamma.rvs(ag,bg,cg, num_mal_workers)
        while np.mean(samples) > avg_labels + 10: 
            samples = stats.gamma.rvs(ag,bg,cg, num_mal_workers)
        
        print(np.mean(samples))
    elif dataset == 'd_sentiment': 
        #number of tasks per mal_workers
        ag,bg,cg   = getDistributionNormalData(ansfile, dataset)
        samples = stats.gamma.rvs(ag,bg,cg , num_mal_workers)
        while np.mean(samples) > avg_labels + 100 or np.mean(samples) < avg_labels - 100  : 
            samples = stats.gamma.rvs(ag,bg,cg , num_mal_workers)
            
        for i,s in enumerate(samples):
            if s>1000:
                samples[i] = 900

        #print(np.mean(samples))
        #print(samples)
    
    
    return samples



def assigntasksperusers(num_adversary, ntasks, numhonestUsers, gamma, dataset, avg_labels, ansfile):
            #assign_task name to id
            #assigned = rndm(1, ntasks, g=0.25, size=int(140*0.01 * numhonestUsers))    
#            assigned = rndm(1, ntasks, g=0.00025, size=int(140*0.01 * numhonestUsers))    
#            
#            assigned = [int(i) for i in assigned]
#            random.shuffle(assigned)
#            
#            num_assigned = assigned[0: num_adversary] #random.sample(list(assigned), num_adversary)
#            
            num_assigned = gen_graph_mal_workers(dataset, num_adversary, avg_labels, ansfile)
            
            taskperadversary = {}
            for i, m in enumerate(num_assigned):
                tasksID = random.sample(range(0, ntasks), int(m))
                adversary_id = 'adv_{}'.format(i)
                taskperadversary[adversary_id] = tasksID  
            
            tid_norm_per_adv={}
            
            for i in range(len(num_assigned)):
                adversary_id = 'adv_{}'.format(i)
                tids = taskperadversary[adversary_id] 
                # fixed and increase the tid in each process
                random.shuffle(tids)
                for idx, gid in enumerate(gamma[1:]):
                    normtasksID = tids[0:int(gid * len(tids))]
                    if adversary_id in tid_norm_per_adv: 
                        tid_norm_per_adv[adversary_id].append(normtasksID)
                    else: 
                        tid_norm_per_adv[adversary_id] = [[]]
                        tid_norm_per_adv[adversary_id].append(normtasksID)
                
            return taskperadversary, tid_norm_per_adv 
    
"""
def preparedAdversayData(dataset_path, truth, ntasks, task_dict, nhonestuser, percentage, gammalist, normbehave, advbehave, iterations, cols, which):
        
        maxpercentage = max(percentage)
        maxad_users = int(maxpercentage * 0.01 * nhonestuser)
        #adv_list = ['adv_{}' for i in range(maxad_users)]
        
        #dirname= dataset_path
        mean = 90 #15
        for niter in range(iterations):
            for gamma in gammalist:
                for norm in normbehave:
                    for adv in advbehave:
                        assignedtasks = assigntasksperusers(maxad_users, mean, ntasks, nhonestuser)
                        adv_answers = poisionAttack(maxad_users,assignedtasks, gamma, norm, adv, truth, task_dict, which)
                        df = pd.DataFrame(adv_answers, columns = cols)
                        #for p in percentage:
        					  #	nrows = int(nhonestuser * 0.01 * p)
                        dirname = dataset_path + "\\iter_{}".format(niter)
                        if not os.path.exists(dirname):
                            os.makedirs(dirname)
                        filename = dirname + "\\g_{}_alpha_{}_beta{}_adalpha{}_adbeta{}.csv".format(gamma, norm[0], norm[1], adv[0], adv[1])
                        df.to_csv(filename, index=False)
"""

def preptaskAssignment(percentage_adv, nhonestuser,ntask, gamma,  which,avg_labels, ansfile ): 
    
#    maxpercentage = max(percentage)
    maxad_users = int(percentage_adv * 0.01 * nhonestuser)

    assignedtasks , tid_norm_per_adv = assigntasksperusers(maxad_users, ntasks, nhonestuser, gamma, which,avg_labels, ansfile )
    
    return assignedtasks , tid_norm_per_adv
                            

def is_reliability_generated(file_path):
    return os.path.exists(file_path)

def loadRealiability(att_adv_reliability_filepath, att_norm_reliability_filepath):
    with open(att_adv_reliability_filepath, 'rb') as input:
        adversayReliability = pickle.load(input)
    with open(att_norm_reliability_filepath, 'rb') as input:
        norm_adversay_behave = pickle.load(input)

    return norm_adversay_behave,  adversayReliability

def generateReliability(num_adversary, att_adv_reliability_filepath, att_norm_reliability_filepath, norm, adv):
    adversayReliability = assign_adversary_reliability(num_adversary,adv, adv)
    norm_adversay_behave = assign_Normal_reliability(num_adversary, norm, norm)
    with open(att_adv_reliability_filepath, 'wb') as output:
        pickle.dump(adversayReliability, output, pickle.HIGHEST_PROTOCOL)

    with open(att_norm_reliability_filepath, 'wb') as output:
        pickle.dump(norm_adversay_behave, output, pickle.HIGHEST_PROTOCOL)

    return  norm_adversay_behave, adversayReliability


def preparedAdversayData(dataset_path, truth, ntasks, task_dict, nhonestuser, percentage, gammalist, normbehave, advbehave, iterations, cols, which,avg_labels_normal, ansfile_normal):
    
    maxpercentage = max(percentage)
    if which == 'product': 
        maxad_users_rel = int(maxpercentage * 0.01 * nhonestuser)
    else:
        maxad_users_rel = int(maxpercentage * 0.01 * nhonestuser)
        
    for perc_adv in percentage:
        maxad_users =  int(perc_adv * 0.01 * nhonestuser)
        for niter in range(iterations):
            
            assignedtasks, tid_norm_per_adv = preptaskAssignment(perc_adv, nhonestuser, ntasks, gammalist, which,avg_labels_normal, ansfile_normal)
            dirname = os.path.join(dataset_path , "untargeted_inj_iteration_{}".format(niter))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
    
            for gidx, gamma in enumerate(gammalist):
                    for norm in normbehave:
                        for adv in advbehave:
                            att_adv_reliability_filepath = os.path.join(dirname , "adv_adv_behave_alpha_{}_beta{}_adalpha{}_adbeta{}.pkl".format(norm[0], norm[1], adv[0], adv[1]))
                            att_norm_reliability_filepath = os.path.join(dirname , "adv_norm_behave_alpha_{}_beta{}_adalpha{}_adbeta{}.pkl".format(norm[0], norm[1], adv[0], adv[1]))
                            if is_reliability_generated(att_adv_reliability_filepath) : 
                                reliability_att_norm , reliability_att_adv =  loadRealiability(att_adv_reliability_filepath,
                                                                                               att_norm_reliability_filepath)
                            else:                            
                                reliability_att_norm , reliability_att_adv = generateReliability(maxad_users_rel, att_adv_reliability_filepath,
                                                                                                 att_norm_reliability_filepath, norm, adv)
                            
    
                            adv_answers = poisionAttack(maxad_users,assignedtasks, gamma, gidx,truth, task_dict, which,
                                                        tid_norm_per_adv, reliability_att_norm , reliability_att_adv  )
                            df = pd.DataFrame(adv_answers, columns = cols)
                            filename = os.path.join(dirname , "g_{}_alpha_{}_beta{}_adalpha{}_adbeta{}_adv{}.csv".format(gamma, norm[0], norm[1], adv[0], adv[1], perc_adv))
                            df.to_csv(filename, index=False)




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


def givIndex(task_dict, ctask, which):
    for i in task_dict:
        if which == 'product':
            if task_dict[i]['question'] == ctask:
                return i
        elif which == 'd_sentiment':
            if str(task_dict[i]['q']) == ctask:
                return i


def modified_assignLabel(adv, adv_alpha, adv_beta,user_tidx, norm_alpha, norm_beta,
                         tasks_dict, which, target_list):              
    # targeted : list of tasks that adversary should behave as adversary
    line = [] 
    user_tidx = [e[0] for e in user_tidx] 
    if len(target_list) != 0 :    
        for ctask in target_list: # tell truth  
            cidx = givIndex(tasks_dict, ctask, which)
            label = getlabelbasedCM(norm_alpha, norm_beta ,tasks_dict[cidx]['truth'] )
            task_name = ctask
            line.append([task_name ,adv, label])

    diff = list(set(user_tidx) - set(target_list))
    for ctask in diff:
        cidx = givIndex(tasks_dict, ctask, which)
        label = getlabelbasedCM(adv_alpha, adv_beta ,tasks_dict[cidx]['truth'] )
        task_name = ctask
        line.append([task_name, adv, label])           
    return line



def ld_writeDicts(filePath,mdict):
    f=open(filePath,'wb')
    newData = pickle.dumps(mdict, 1)
    f.write(newData)
    f.close()
    
    
    
    #############
    
    # read file decoding with cPickle/pickle (as binary)
def ld_readDicts(filePath):
    f=open(filePath,'rb')
    data = pickle.load(f)
    f.close()
    return data
    
    # return dict data to new dict
    #newDataDict = ld_readDicts('C:/Users/Lee/Desktop/test2.dta')
    
def assignHonestTasksPerGamma(gammalist, user_tasks, path):
    dg = {}
    for u in user_tasks: 
        dg[u] = {}
        for g in gammalist: 
            sample_size = round(g * len(user_tasks[u]))
            if sample_size < len(user_tasks[u]):
                sample = user_tasks[u][0:sample_size]
                t = [s[0] for s in sample]
                dg[u][g] = t
            else: 
                dg[u][g] = [s[0] for s in user_tasks[u]]
    ld_writeDicts(path,dg)
    return dg
              

def poisionRecords(adv_list, assignedTask,adversayReliability,
                   norm_adversay_behave, tasksdict,which,target_list,user_dict): 
    advAnswers = []
    # check is it really relate to the specific one or not
    # adv_list is the number of all users
    for i in range(adv_list):
        u = user_dict[i]
        #target_list = diff()
        #print("it is working")
        #print(i)
        advAnswers += modified_assignLabel(u, adversayReliability.TTcdf[i],adversayReliability.FFcdf[i],
                                  assignedTask[u],norm_adversay_behave.TTcdf[i],
                                  norm_adversay_behave.FFcdf[i], tasksdict, which, target_list[u])
                
    return advAnswers



def modifiesRecords(percentage, nhonestuser,ntasks, gammalist, iterations, dataset_path,
                   truth, task_dict, cols, which, user_dict, datafile, honestFile):
    maxad_users= nhonestuser   
    e2wl,w2el,label_set = gete2wlandw2el(datafile)
    
    if os.path.exists(honestFile):
        honestTasks_users_gamma = ld_readDicts(honestFile)
    else:
        honestTasks_users_gamma = assignHonestTasksPerGamma(gammalist, w2el, honestFile)
    
    for niter in range(iterations):
        dirname = dataset_path + "\\modify_untargeted_iteration_{}".format(niter)
        if not os.path.exists(dirname):
            os.makedirs(dirname)


    assignedtasks = w2el
    
    for gidx, gamma in enumerate(gammalist):
        honest_taks_users = {}
        for u in honestTasks_users_gamma:
            honest_taks_users[u] = honestTasks_users_gamma[u][gamma]
        for norm in normbehave:
            for adv in advbehave:
                att_adv_reliability_filepath = dirname + "\\adv_adv_behave_alpha_{}_beta{}_adalpha{}_adbeta{}.pkl".format(norm[0], norm[1], adv[0], adv[1])
                att_norm_reliability_filepath = dirname + "\\adv_norm_behave_alpha_{}_beta{}_adalpha{}_adbeta{}.pkl".format(norm[0], norm[1], adv[0], adv[1])
                if is_reliability_generated(att_adv_reliability_filepath) : 
                    reliability_att_norm, reliability_att_adv = loadRealiability(att_adv_reliability_filepath,
                                                                                  att_norm_reliability_filepath)
                else:                            
                    reliability_att_norm, reliability_att_adv = generateReliability(maxad_users,
                                                                                    att_adv_reliability_filepath, 
                                                                                    att_norm_reliability_filepath,
                                                                                    norm, adv)
                
                
                adv_answers = poisionRecords(maxad_users,assignedtasks,reliability_att_adv,reliability_att_norm,
                                                     task_dict,which,honest_taks_users, user_dict)
                
                df = pd.DataFrame(adv_answers, columns = cols)
                filename = dirname + "\\g_{}_alpha_{}_beta{}_adalpha{}_adbeta{}.csv".format(gamma, norm[0], norm[1], adv[0], adv[1])
                df.to_csv(filename, index=False)    
    return


def getcolumn_ansFile( path):
        with open(path, "r") as f:
            col = next(csv.reader(f))
        return col

def rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)


def getMultiplyNums(value):
        value = 240
        N = math.ceil(math.sqrt(value))        
        while value % N != 0:
                N +=1
        M = value / N 
        return N, M							

def readfilegetcol( datapath, colname):
            columns = defaultdict(list) # each value in each column is appended to a list
            with open(datapath) as f:
                reader = csv.DictReader(f) # read rows into a dictionary format
                for row in reader: # read a row as {column1: value1, column2: value2,...}
                    for (k,v) in row.items(): # go over each column name and value 
                        columns[k].append(v) # append the value into the appropriate list
                                             # based on column name k
            return list(set(columns[colname]))
    


if __name__ == "__main__":
    
    ispartial = False
    type_attack = 'j' # inject or modified
    whichdataset= 'product' #'product' #'product' #'d_sentiment'
    #whichdataset= 'd_sentiment' #'product' #'product' #'d_sentiment'
    
    if whichdataset == 'product':
        avg_label_normal = 141
    elif whichdataset == 'd_sentiment':
        avg_label_normal = 255
        
    datadir = os.path.join(os.getcwd(), 'realdataset', whichdataset)
    ansfile =  os.path.join(datadir ,'answers.csv')
    truthfile = os.path.join( datadir , 'truth.csv')
    if whichdataset == 'product':
        tasksList = readfilegetcol(truthfile, 'question')
    elif whichdataset == 'd_sentiment':
        tasksList = readfilegetcol(truthfile, 'q')

    honest_users_List = readfilegetcol(ansfile,'worker')
    ntasks = len(tasksList)
        
    numhonestUsers = len(honest_users_List )
    df = pd.read_csv(truthfile)
    true_labels = df['truth']
    tasksdict = df.to_dict('index')
    
    e2wl,w2el,label_set = gete2wlandw2el(ansfile)    
    users_dict ={}
    numhonestUsers = len(honest_users_List )
    k_list = list(w2el.keys())
    for i, u  in enumerate(k_list):
        users_dict[i] = k_list[i]


    
    columns = getcolumn_ansFile(ansfile)
    if type_attack == 'm' : 
        percentage= [3,5, 10, 20, 30, 40, 50, 60, 70, 100]
        gammalist = [ 0.3, 0.4, 0.5, 0.6 , 0.7]
    else:
        #percentage = [80,90]
        percentage=[20,30,40,50] # [20, 30, 40, 50, 60, 70,100, 140]
        #percentage= [49,52,53,54,55]
        
        gammalist = [0] #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 , 0.7]
    
    #normbehave = [(0.85, 0.65)] #[1.0, 0.8, 0.6] [0.9,0.9]
    #advbehave = [(0.2, 0.0)]  #[0.0, 0.2, 0.3, 0.4]  [0.1,0.15]
    
    if whichdataset == 'product':
        # Just for validate the svd 
        normbehave = [(1.0, 1.0)]#, (0.85, 0.70)]
        advbehave = [(0.0, 0.8), (0.0, 0.9), (0.0, 1.0), (1.0, 0.0),(0.8, 0.0), (0.9,0.0)]#[(0.0,0.1), (0.0,0.3), (0.1,0), (0.3,0), (0, 0.2)] 
#        advbehave = [(0.0,0.1), (0.0,0.3), (0.1,0), (0.3,0), (0, 0.2)] 

		#advbehave = [(0.0,0.0)]#, (0.2,0.0), (0.2,0.1)] 

        #normbehave = [(1.0, 1.0), (0.95, 0.75), (0.85, 0.65), (0.9, 0.6), (0.75, 0.75)] #[1.0, 0.8, 0.6] [0.9,0.9]
        #advbehave = [(0.0,0.0),  (0.2,0.1), (0.2, 0.0), (0.3, 0.0), (0.5, 0.0)] 
    elif whichdataset == 'd_sentiment':
        normbehave = [(1.0, 1.0)]#, (0.9, 0.9), (0.85, 0.85), (0.8, 0.8), (0.75, 0.75), (0.7, 0.7)] #[1.0, 0.8, 0.6] [0.9,0.9]
        advbehave = [(0.0, 0.8), (0.0, 0.9) , (0.8, 0.0), (0.4,0.0), (0.2, 0.0)]#[(0.0,0.1), (0.0,0.3), (0.1,0), (0.3,0), (0, 0.2)] 
#advbehave = [(0.0,0.0), (0.1,0.1)]#, (0.1,0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)] 

    iters = 10
     
    if type_attack == 'j':
        preparedAdversayData(datadir, true_labels, len(tasksList), tasksdict , numhonestUsers, percentage, gammalist, normbehave, advbehave, iters, columns , whichdataset, avg_label_normal, ansfile)
    elif type_attack == 'm' : 
        honestfile =  datadir + '\\honest.dta'
        modifiesRecords(percentage, numhonestUsers,ntasks, gammalist, iters, datadir,
                   true_labels, tasksdict, columns, whichdataset,users_dict,ansfile, honestfile)

    
    
    
    
    


