# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:03:01 2018

@author: tahma
"""
from scipy.stats import beta
import random 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import  os, pickle
import confusion_matrix as cm
from shutil import copyfile
from numpy.random import choice
from pathlib import Path, PureWindowsPath


from scipy.stats import powerlaw, norm, gamma, beta
from collections import Counter
import seaborn as sns




def fib(n):
    if n == 0:
        return [0]
    elif n == 1:
        return [0, 1]
    else:
        lst = fib(n-1)
        lst.append(lst[-1] + lst[-2])
        return lst


def assignLabletoTask(prob_gt, num_tasks):
    # TODO:Should it be based on bernoulli or I should exactly assign the 50% of the task as +1???    
     nums = np.ones(num_tasks)   
     nums.astype(int)
     nums[:int(num_tasks*(1-prob_gt))] = 0
     np.random.shuffle(nums)
     return nums
     #return bernoulli.rvs(prob_gt, size=num_tasks)


def getDistributionNormalData(normal_data, dataset):
    normal_data = os.path.join(os.getcwd(), 'realdataset', 'product', 'answers.csv')
    df = pd.read_csv(normal_data)
    #ax = df.plot.hist(bins=12, alpha=0.5)
    df.hist(column='answer', bins = 200)
    freq = df.groupby('worker').count()
    freq = freq.sort_values(by=['question'])
    max_freq = max(freq['question'])
    min_freq = min(freq['question'])
    
    freq.hist(column = 'question', bins = (max_freq-min_freq)//100)
    
    from scipy import stats  
    import numpy as np  
    import matplotlib.pylab as plt

    num_bins = (max_freq-min_freq)//75
    # plot normed histogram
    plt.hist(freq['question'], normed=True, bins= num_bins)

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
    plt.plot(lnspc, pdf_gamma, label="Gamma")
    
     # exactly same as above
#    ag,bg,cg = stats.powerlaw.fit(freq['question'])  
#    pdf_powerlaw = stats.powerlaw.pdf(lnspc, ag, bg,cg)  
#    plt.plot(lnspc, pdf_powerlaw, label="POWERLAW")

#    # guess what :) 
#    ab,bb,cb,db = stats.beta.fit(freq['question'])  
#    pdf_beta = stats.beta.pdf(lnspc, ab, bb,cb, db)  
#    plt.plot(lnspc, pdf_beta, label="Beta")

    plt.show()  
    
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
    ag,bg,cg = getDistributionNormalData(ansfile, dataset)
    if dataset == 'product': 
        #number of tasks per mal_workers
        
        samples = gamma.rvs(ag,bg,cg, num_mal_workers)
        while np.mean(samples) > avg_labels + 10: 
            samples = gamma.rvs(ag,bg,cg, num_mal_workers)
        
        print(np.mean(samples))
    
    return samples
def rndm(a, b, g, size=1):
   """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
   r = np.random.random(size=size)
   ag, bg = a**g, b**g
   return (ag + (bg - ag)*r)**(1./g)

def generate_graph(num_user, num_tasks, dense):
    aseq = []
    xx = rndm(1, num_user+1, g=-0.75, size= num_tasks * (dense + 1))
    for i in xx: 
        if i > num_tasks:
            aseq.append(num_tasks-1)
        else:
            aseq.append(int(i))
            
    freq = Counter(aseq)
    ans = []

    ch = [i for i in range(num_user)]
    prob = [ freq[i+1]/sum(freq.values()) for i in range(num_user)]
#    non_zero = len(prob)-prob.count(0.0)
#    if dense > non_zero:
#        max_prob = max(prob)
#        rnd_modify = dense - non_zero
#        fix_val = max_prob / rnd_modify
#        id_max = prob.index(max_prob)
#        id_zeros = [i for i, x in enumerate(prob) if x == 0.0]
#        
#        prob[id_max] = 0
#        for idx in id_zeros[:rnd_modify+1]: 
#            prob[idx] = fix_val
#        max_prob = max(prob)
#        id_max = prob.index(max_prob)
#        
#        if sum(prob) > 1: 
#            prob[id_max] += (1-sum(prob))
        
        
        
        
#    for i,p in enumerate(prob):
#        if p == 0.0:
#            prob[i] = 0.001
     
    for i in range(num_tasks):
        draw = choice(ch , dense, p=prob,  replace=False)
        ans.append(list(draw))

    cnt= [0 for i in range(num_user)]
    for i in range(num_tasks):
        for  j in range(dense):
            cnt[ans[i][j]] +=1
            
    bseq = [dense for i in range(num_tasks)]
    
    gw = nx.bipartite.generators.configuration_model(cnt,bseq)
    pos = nx.spring_layout(gw)
    nx.draw_networkx(gw, pos)
    #plt.show()
    return gw
    
        



    
    

def old_generate_graph(num_user, num_tasks):
        """
        # vriable : powerlaw
        power = 2.5#2.5
        aseq = nx.utils.powerlaw_sequence(num_user,power) 
        aseq = [int(i) for i in aseq]
        prob = [0.5] 
        for pr in prob:        
            gw = nx.bipartite.preferential_attachment_graph(aseq,pr, None, None)
         
        createagain = True

        power = 1.7 #2.5
        var1 = 0
        ratio = int(num_tasks/ num_user)
        while createagain:           
            aseq = nx.utils.powerlaw_sequence(num_tasks,power) 
            aseq = [int(i) for i in aseq]
            var1 = np.var(aseq)
            if  var1 > (num_user)//2 and var1 <= num_user:
                createagain = False
            else:
                createagain = True
        
        bseq = [0] * num_user
        fibseq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89,34, 21,13, 8, 5, 3, 2, 1,1]
        #fibseq = fibseq[1:]

    
        idx_aseq = 0
        for i,f in enumerate(fibseq):
            bseq[i] = sum(aseq[idx_aseq:idx_aseq+f])
            idx_aseq += f
        if idx_aseq < num_tasks: 
            bseq[-1] += sum(aseq[idx_aseq:])
        
        """
        aseq = [107, 2, 3, 21, 6, 31, 1, 2, 400, 1, 164, 14, 199, 2, 1, 187, 7, 10, 6, 67] # p = 1.4
        
        bseq = [2, 4, 1, 1, 1, 1, 1, 7, 1, 5, 1, 1, 1, 1, 2, 1, 4, 2, 1, 1, 1, 1, 1, 2, 1, 2, 4, 1, 1, 1, 20, 16, 1, 1, 16, 3, 4, 2, 3, 1,1, 1, 1, 1, 1, 6, 10, 3, 1, 1, 3, 2, 3, 3, 1, 3, 1, 2, 14, 2, 1, 2, 1, 1, 1, 2, 1, 8, 3, 6, 1, 1, 1, 3, 1, 8, 2, 1, 1, 1,1, 2, 2, 2, 3, 1, 1, 2, 1, 13, 5, 2, 1, 1, 2, 2, 3, 2, 3, 3, 5, 2, 1, 1, 3, 3, 6, 1, 20, 1, 2, 2, 1, 2, 1, 4, 3, 2, 1, 3, 4, 2, 4, 2, 3, 1, 2, 1, 3, 2, 1, 1, 1, 4, 2, 6, 4, 1, 1, 2, 3, 1, 2, 3, 13, 2, 1, 2, 4, 10, 2, 2, 1, 1, 2, 1, 2, 1, 7, 2,7, 1, 18, 2, 2, 1, 1, 2, 2, 1, 1, 4, 2, 1, 2, 3, 1, 1, 1, 1, 6, 1, 1, 1, 2, 1, 2, 1, 3, 1, 8, 2, 2, 1, 3, 2, 1, 13, 2, 3, 5, 1, 5, 1, 1, 1, 1, 1, 14, 1, 2, 6, 2, 2, 2, 2, 2, 2, 1, 1, 2, 3, 2, 4, 4, 10, 1, 2, 9, 3, 1, 1, 11, 1, 11, 1, 1, 1, 2, 3, 20, 3, 1, 2, 2, 2, 5, 1, 1, 1, 4, 2, 2, 3, 2, 1, 1, 11, 1, 2, 2, 3, 1, 2, 6, 2, 19, 2, 1, 1, 3, 1, 18, 16, 1, 2, 12, 1, 1, 8, 2, 2, 6, 10, 3, 20, 3, 15, 2, 1, 1, 1, 2, 3, 2, 5, 1, 3, 1, 2, 1, 1, 1, 4, 9, 1, 3, 4, 3, 4, 3, 2, 4, 2, 2, 4, 5, 3, 2, 2, 3, 4, 3, 2, 2, 8, 3, 2, 3, 3, 20, 3, 1, 1, 2, 2, 1, 1, 3, 1, 3, 14, 3, 3, 8, 3, 2, 3, 17, 1, 1, 1, 5, 2, 8, 1, 1, 1, 1, 3,1, 1, 1, 4, 2, 4, 2, 6, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 3, 3, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 2, 5, 1, 1, 1, 1, 10 ]
        
        #bseq = [sum(aseq[ratio*(i-1):ratio*i]) for i in range(1,num_tasks+1)]
        #bseq = bseq[0:num_user]
        """
        aseq = [1, 1, 2, 1, 2, 5, 3, 313, 3, 400, 1, 4, 1, 1, 6, 1, 1, 4, 2, 1] #p = 2.5
        bseq = [2, 5, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 3, 3, 1, 1, 1, 3, 4, 2, 2, 1, 2,1, 1, 1, 1, 2, 6, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 4, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 4, 2, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 1, 3, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 3, 2, 3, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2,4, 2, 2, 1, 2, 2, 1, 1, 1, 3, 1, 1, 2, 1, 1, 2, 2, 1, 4, 2,1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 3, 6, 2, 2, 3, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 3, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 3, 3, 2, 1, 2, 1, 2,1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1,1, 1, 4, 2, 10, 1, 1, 1, 2, 1, 3, 2, 1, 2, 1, 1, 1, 1, 1, 5,8, 3, 2, 1, 3, 2, 2, 5, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 3, 1, 1, 1, 3, 2, 2, 1, 1, 1, 4, 1, 1, 1, 14, 2, 2, 1, 1, 1, 2, 5, 1, 1, 2, 4, 2, 33, 1, 1, 1, 1, 2, 2, 1, 3, 1, 1, 2, 1, 1, 2, 2, 1, 4, 2, 1, 2, 4,3, 4, 1, 1, 1, 3, 1, 1, 1, 7, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2,2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 6, 1, 1, 8, 2, 1, 4,1, 1, 3, 3, 3, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1,18, 1, 2, 2, 1, 1, 2, 1, 2, 3, 5, 1, 2, 9, 5, 2, 1, 1, 2, 1] # p = 3
        """
        
        gw = nx.bipartite.generators.configuration_model(aseq,bseq)
        #for i in range(num_user):
        #    print (gw.degree(i))
        pos = nx.spring_layout(gw)
        nx.draw_networkx(gw, pos)
        plt.show()
        return gw

def assignhonestyLabel(prob, nu):
        wlist = []
        for i in range(nu): # honest == 1 and adversary = 0
            r = random.random()
            wlist.append('honest') if r <  prob else wlist.append('adversary')
        return wlist

def confusion_matrix_AssignReliability(wlist, avg_alpha,svd_alpha ,avg_beta, svd_beta):
    #labels = [0,1]    
    user_CMatrix = cm.Confusion_Matrix()
    user_CMatrix.createCM(len(wlist), avg_alpha,svd_alpha ,avg_beta, svd_beta)
    worker = pd.DataFrame()
    cmatrix = []
    for i in range(len(wlist)):
        cmatrix.append(np.zeros([2, 2])) 
    #user_CMatrix.setreliabilitybasedCDF(distalpha, distbeta)
        
    print("Generate CDF distribution for honest people reliability ")
    worker["reliability"] = np.NAN
    worker["cdf"] = np.NAN 
    worker["cdf_one"] = np.NaN
    worker["cdf_zer0"] = np.NaN
    worker["reliability_one"] = np.NaN
    worker["reliability_zero"] = np.NaN

    worker["reliability_one"] = user_CMatrix.TT
    worker["reliability_zero"] =  user_CMatrix.FF

    worker["cdf_one"] = user_CMatrix.TT  #distalpha.cdf(worker["reliability_one"])
    worker["cdf_zero"]  =  user_CMatrix.FF #distbeta.cdf(worker["reliability_zero"])
       
    for i, val in enumerate(worker["cdf_zero"]):
        row = [val, 1-val]
        cmatrix[i][0] = row
    for i, val in enumerate(worker["cdf_one"]):
        row = [1-val, val]
        cmatrix[i][1] = row
    


    #worker["cdf"].fillna(0.1, inplace = True) 
    return worker["reliability_one"], worker["reliability_zero"],worker["cdf_one"],worker["cdf_zero"], cmatrix
    #print(worker.cdf.mean())


def adv_model_AssignReliability(wlist, avg_alpha,svd_alpha ,avg_beta, svd_beta):
        user_CMatrix = cm.Confusion_Matrix()
        user_CMatrix.createCM(len(wlist), avg_alpha,svd_alpha ,avg_beta, svd_beta)
        return user_CMatrix
                


def model_AssignReliability(wlist, reliabilityMode, fixed_reliability, avg_alpha,svd_alpha ,avg_beta, svd_beta):
        worker = pd.DataFrame()
        confMtrix_users = []
        for i in range(len(wlist)):
            confMtrix_users.append(np.zeros([2, 2])) 

        se = pd.Series(wlist)
        worker['w_identity'] = se.values
        worker['wid'] = worker.index
        worker = worker[['wid','w_identity']]
                    
        honest_users = len(worker[worker['w_identity'] == 'honest'])
        adversarial_users = len(worker[worker['w_identity'] == 'adversary'])
        print("honest: , adversry: ",honest_users, adversarial_users)
        
        if reliabilityMode == 'CM':
            
            worker["reliability"] = np.NAN
            worker["cdf"] = np.NAN 
            worker["cdf_one"] = np.NaN
            worker["cdf_zer0"] = np.NaN
            worker["reliability_one"] = np.NaN
            worker["reliability_zero"] = np.NaN
            worker["reliability_one"], worker["reliability_zero"],worker["cdf_one"],worker["cdf_zero"], confMtrix_users =  confusion_matrix_AssignReliability(wlist, avg_alpha,svd_alpha ,avg_beta, svd_beta)
            print("Average Normal workers reliability alpha  :")
            print(worker.cdf.mean())
            print("Average Normal workers reliability Beta :")
            print(worker.cdf.mean())        
            

        elif reliabilityMode == 'prob':
             print("Generate CDF distribution for honest people reliability ")
             alpha_values = [1, 1.5, 3.0, 0.2]
             beta_values = [1, 1.5, 3.0, 3.5]
             #Generate random numbers:
             dist = beta(alpha_values[3], beta_values[3])
             worker["reliability"] = np.NAN
             worker["cdf"] = np.NAN                   
             worker["reliability"].iloc[worker[worker["w_identity"] == 'honest'].index] =  dist.rvs( size =  honest_users) # transformed_rand
             worker["cdf"].iloc[worker[worker["w_identity"] == 'honest'].index] =  dist.cdf(worker["reliability"][worker["w_identity"] == 'honest'])
             worker["cdf"].fillna(0.1, inplace = True) 
             print("Average Normal workers reliability  :")
             print(worker.cdf.mean())
             #print("Average Normal workers reliability Beta :")
             #print(worker.cdf.mean())

        elif reliabilityMode == 'fixed':
             worker["cdf"] = fixed_reliability 
            
        
        return worker, confMtrix_users

def getThreshhold(num_users, mean_cdf, set_edges):
    #get_graph_edges
    threshhold =  (len(set_edges) / (num_users*num_users))  * mean_cdf#  ah_workers['cdf'].mean() # 14% 
    threshhold = '%.6f'%threshhold
    threshhold = float(threshhold)
    return threshhold

def savegraph(graph, filepath):
    nx.write_gml(graph,filepath)

def loadgraph(filepath): 
    return  nx.read_gml(filepath)


def write_to_file(df, dirname, fname ):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        path = dirname +'\\'+fname
        file_path = os.path.join(os.getcwd(), path )

        try:
            os.remove(file_path)
        except OSError:
            pass
        
        df.to_csv(file_path, sep=',', encoding='utf-8',index=False)
        
        return


# Lable is 0 and 1
def findMaximumProbableAnswer(confusionMatrix, truth):
    prob = [confusionMatrix[0][truth] /  confusionMatrix[truth][truth] , confusionMatrix[1][truth] /  confusionMatrix[truth][truth]]
    index, value = max(prob, key=lambda item: item[1])
    return index
 
    
def getlabelbasedCM(alpha, beta, truth):  
    if truth == 1.0 :
       if random.randrange(1,100) in range(1,int(alpha*100)):
          return 1
       else: return 0
    else:
       if random.randrange(1,100) in range(1,int(beta*100)):
          return 0
       else: 
          return 1      


def load_userBehavior(mean_alpha,mean_beta, graphDir,mean_a_adv,mean_b_adv, behave_gamma):

    src = graphDir +'\\user_df_a{}_b{}.csv'.format(mean_alpha, mean_beta)    
    dst = graphDir +'\\user_df_a{}_b{}_aprim{}_bprim{}_gamma{}_p{}.csv'.format(mean_alpha, mean_beta,mean_a_adv, mean_b_adv, behave_gamma, 0)
    if os.path.isfile(dst) == False:
        copyfile(src, dst)
    
    src = graphDir +'\\workers_df_a{}_b{}.csv'.format(mean_alpha, mean_beta)    
    dst = graphDir +'\\workers_df_a{}_b{}_aprim{}_bprim{}_gamma{}_p{}.csv'.format(mean_alpha, mean_beta,mean_a_adv, mean_b_adv, behave_gamma, 0)
    if os.path.isfile(dst) == False:
        copyfile(src, dst)


    src = graphDir +'\\answers_a{}_b{}.csv'.format(mean_alpha, mean_beta)    
    dst = graphDir +'\\answers_a{}_b{}_aprim{}_bprim{}_gamma{}_p{}.csv'.format(mean_alpha, mean_beta,mean_a_adv, mean_b_adv, behave_gamma, 0)
    if os.path.isfile(dst) == False:
        copyfile(src, dst)
        


def correctPath_os(file):
    filename = PureWindowsPath(file)
    correct_path = Path(filename)

    return correct_path

def gen_userBehavior(num_users,num_tasks, reliabilityModel, fixed_reliability,
                        mean_alpha, svd_alpha, mean_beta,
                        svd_beta, graphDir,tasks_truthLabel, assignedTaskperWorker, mean_a_adv,
                        mean_b_adv, behave_gamma, att_adv_reliability_filepath):
    
                   
    print("Model users with confusion matrix")
    ahlist = []
    prob_honestusers = 1
    ahlist =  assignhonestyLabel(prob_honestusers, num_users)
    conf_matrix_list = []   
    ah_workers = pd.DataFrame()
    ah_workers , conf_matrix_list = model_AssignReliability(ahlist, reliabilityModel, fixed_reliability, mean_alpha, svd_alpha, mean_beta, svd_beta)
    
    
    if os.path.exists(correctPath_os(att_adv_reliability_filepath)) == False:
            adv_conf_matrix_list = adv_model_AssignReliability(ahlist,  mean_a_adv, svd_alpha, mean_b_adv, svd_beta)
            with open(att_adv_reliability_filepath, 'wb') as output:
                pickle.dump(adv_conf_matrix_list, output, pickle.HIGHEST_PROTOCOL)
    
    t_Lables = []
    for x in range(0,num_tasks):
        t_Lables.append("task_{0}".format(x))    
    
    ahwt_df = pd.DataFrame(index = ah_workers.index , columns = t_Lables)
    """
    adv_idx = list(ah_workers[ah_workers['w_identity'] =='adversary'].index)
    for aid in adv_idx: 
         for taskidx in assignedTaskperWorker[aid]:
             ahwt_df.iloc[aid , taskidx] = 1
    """
    ah_honest_idx = list(ah_workers[ah_workers['w_identity'] =='honest'].index)
    for uid in ah_honest_idx:
        if assignedTaskperWorker.get( uid, None ) == None:
            continue
        else:
            for taskidx in assignedTaskperWorker[uid]:
                if reliabilityModel == 'CM':    
                    alpha = conf_matrix_list[uid][1][1]
                    beta = conf_matrix_list[uid][0][0]
                    ahwt_df.iloc[uid , taskidx] = getlabelbasedCM(alpha, beta, tasks_truthLabel[taskidx]) # np.random.choice(2,p = pr)   #np.argmax(conf_matrix_list[uid][tasks_truthLabel[taskidx]]) #findMaximumProbableAnswer(conf_matrix_list[uid], tasks_truthLabel[taskidx])               
    
    answer = pd.DataFrame(columns=['question','worker','answer'])
    num_line = 0
    for t in range(num_tasks):
        for u in range(num_users):
            if np.isnan(ahwt_df.loc[u][t]) == False: 
                line = ['task_'+str(t), 'user_'+str(u), int(ahwt_df.loc[u][t])]
                answer.loc[num_line] = line
                num_line += 1
    
    #threshold = getThreshhold(num_users, ah_workers.cdf.mean(), set(gwt.edges()))
    #write_to_file(ah_workers, 'data','workers_df.csv') 
    write_to_file(answer, 'data','answers.csv') 
    
	#remain_name = '_a{}_b{}_aprim{}_bprim{}_gamma{}_p{}.txt'.format(mean_a_norm, mean_b_norm,mean_a_adv, mean_b_adv, behave_gamma,0)

    #write_to_file(answer, graphDir,'answers_a{}_b{}_aprim{}_bprim{}_gamma{}_p{}.csv'.format(mean_alpha, mean_beta,mean_a_adv,mean_b_adv, behave_gamma, 0))
    write_to_file(answer, graphDir,'answers_a{}_b{}.csv'.format(mean_alpha, mean_beta)) 
    # P means percentage of attacker  
    write_to_file( answer, graphDir , 'answers.csv')       
    #write_to_file(ah_workers, graphDir,'workers_df_a{}_b{}_aprim{}_bprim{}_gamma{}_p{}.csv'.format(mean_alpha, mean_beta,mean_a_adv, mean_b_adv, behave_gamma, 0)) 
    write_to_file(ah_workers, graphDir,'workers_df_a{}_b{}.csv'.format(mean_alpha, mean_beta)) 
     

    path = graphDir +'\\user_df_a{}_b{}.csv'.format(mean_alpha, mean_beta)
    ahwt_df.to_csv(path, sep=',', encoding='utf-8',index=False)


    """
    path = graphDir +'\\user_df.csv' #.format(mean_alpha, mean_beta, 0)
    file_path = os.path.join(os.getcwd(), path )
    if os.path.isfile(file_path) == False: 
        ahwt_df.to_csv(file_path, sep=',', encoding='utf-8',index=False)
    """
    
    return ahwt_df
 

def gen_user_task_Graph(num_users,num_tasks, 
                        tasks_lable_proportion,graphDir): 
    
    w_Lables = []
    for x in range(0,num_users):
        w_Lables.append("user_{0}".format(x))
    print("Create Worker_task assignment graph")
    gwt = generate_graph(num_users, num_tasks, 3)
    #num_tasks = len(nx.nodes(gwt)) - nu
    t_Lables = []
    for x in range(0,num_tasks):
        t_Lables.append("task_{0}".format(x))
                    
    conectivity = nx.edges(gwt)
    
    print("Graph is created!")    
    assignedTaskperWorker = dict()
    for w, t in conectivity:
        if w in assignedTaskperWorker:
               assignedTaskperWorker[w].append(t - num_users)
        else:
               assignedTaskperWorker[w] = [t - num_users]
            
    print("Assign the truth label +1 , 0 to each tasks based on .............?????")
    tasks_truthLabel = assignLabletoTask(tasks_lable_proportion, num_tasks)
    #p_index = list(np.where(tasks_truthLabel==1)[0])
    #n_index= list(np.where(tasks_truthLabel== 0)[0])
    truth = pd.DataFrame(columns=['question','result'])
    num_line = 0

    for t in range(num_tasks):
        line = ['task_'+str(t), tasks_truthLabel[t]]
        truth.loc[num_line] = line
        num_line += 1

    write_to_file(truth, graphDir , 'truth.csv')    
    np.save(graphDir + '\\taskperworkerDict.npy', assignedTaskperWorker) 
    return tasks_truthLabel, assignedTaskperWorker                

