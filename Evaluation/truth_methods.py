# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 14:26:05 2017

@author: tahma
"""
import pandas as pd
import numpy as np
from methods.c_MV import method as OMV
from methods.c_EM import method as em
from methods.LFCbinary import method as lfc
from methods.l_KOS import method as kos
from methods.bcc import method as bcc
from methods.MVSoft import method as mvsoft
from methods.MVHard import method as mvhard
import matplotlib.pyplot as plt
from methods.l_ZenCrowd import method as zc
from pathlib import Path, PureWindowsPath

#from My_Methods import f_EM as fem
import os , shutil, csv

def creatw2cm(file):
    with open(file) as f:
        lines = f.readlines()
    w2cm = {}
    label_set = ['0', '1']
    length = len(lines) - 1
    
    i = 6
    while i < length : 
        worker = lines[i].strip() 
        w2cm[worker] = {}
        for tlabel in label_set:
            w2cm[worker][tlabel] = {}
            if tlabel == '0':
                    index = i +2
            else:
                    index = i+ 3
            label = lines[index].split(',')[1:]
            w2cm[worker][tlabel]['0'] = label[0]
            w2cm[worker][tlabel]['1'] = label[1]
        i = i+4
    return w2cm
                
    
    

def correctPath_os(file):
    filename = PureWindowsPath(file)
    correct_path = Path(filename)

    return correct_path


def perf_measure(true_labels, pred_labels):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
     
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
     
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
     
    return (TP,FP,TN,FN)


# For syntethic data        
def applyTruthInferenceMethods(num_adv, datafile, n_iter,true_labels, truthfile, graphDir, average_cm, gamma):
    change_adver_accu = pd.DataFrame(columns = ['num_attackers', 'accuracy', 'method', 'iter'])

    #rates = pd.DataFrame(columns = ['num_attackers','method', 'iter', 'TPR', 'FPR', 'TNR', 'FNR'])

    #change_FScore = pd.DataFrame(columns = ['num_attackers', 'fscore', 'method', 'iter'])    
    
    models = ['mv','em','mvsoft','mvhard', 'bcc', 'lfc', 'kos', 'zc']#, 'kos','zen']
    subdirectory = "a{}_b{}_aa{}_ab{}_g{}_p{}".format(average_cm[0], average_cm[1],average_cm[2], average_cm[3],gamma, num_adv) 
    if not os.path.exists(correctPath_os(graphDir +"\\"+ subdirectory)):
        os.makedirs(correctPath_os(graphDir +"\\"+ subdirectory))
    for method in models:
	
        if method == 'mv':
                 print('assessing MV method')
                 e2wl,w2el,label_set = OMV.gete2wlandw2el(datafile)
                 e2lpd = OMV.MV(e2wl,w2el,label_set).Run()
                 mvacc = OMV.getaccuracy(truthfile, e2lpd, label_set)                  
                 change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, mvacc , 'mv', n_iter]
                 
        elif method == 'mvsoft':
                print('assessing MVSOFT method')
                """
                e2wl,w2el,label_set = mvsoft.gete2wlandw2el(datafile)
                firstacc, bestacc, index, removedworkers = mvsoft.MVSoft(e2wl,w2el,label_set, truthfile).Run()
                if index != 0:
                    credfile = correctPath_os(graphDir +'\\'+ subdirectory+'\\mvsoft_credibility{}.csv'.format(num_adv))
                    with open(credfile, "w") as output:
                        writer = csv.writer(output, lineterminator='\n')
                        for val in removedworkers:
                            writer.writerow([val])  
                            
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, bestacc , 'mvsoft', n_iter]
                """
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, 0 , 'mvsoft', n_iter]
                
        elif method == 'mvhard':
                #e2wl,w2el,label_set = mvhard.gete2wlandw2el(datafile)
                #firstacc, bestacc, index = mvhard.MVHard(e2wl,w2el,label_set, truthfile).Run()
                bestacc = 0.0
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, bestacc , 'mvhard', n_iter]

        elif method == 'em':
                print('assessing EM method')
                workermodel = 'cm'
                e2wl,w2el,label_set = em.gete2wlandw2el(datafile) # generate structures to pass into EM
                iterations = 20 # EM iteration number
                initquality = 0.7
                e2lpd, w2cm = em.EM(e2wl,w2el,label_set,initquality,workermodel, datafile).Run(iterations)
                userCredit = getRelability(w2cm)
                
                credfile = graphDir +'\\'+ subdirectory +'\\em_credibility{}.csv'.format(num_adv)
                w = csv.writer(open(credfile, "w"))
                for key, val in w2cm.items():
                   w.writerow([key, val])
                showrealiability(userCredit)                
                accuracy = em.getaccuracy(truthfile, e2lpd, label_set)
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, accuracy, 'em', n_iter]
        elif method == 'lfc':
                print('assessing LFC method')
                """
                workermodel = 'cm'
                e2wl,w2el,label_set = lfc.gete2wlandw2el(datafile) # generate structures to pass into EM
                iterations = 20 # EM iteration number
                initquality = 0.7
                e2lpd, w2cm = lfc.EM(e2wl,w2el,label_set,initquality,workermodel, datafile).Run(iterations)
                #print (w2cm)    
                userCredit = getRelability(w2cm)
                
                credfile = graphDir +'\\'+ subdirectory+'\\LFC_credibility{}.csv'.format(num_adv)
                w = csv.writer(open(credfile, "w"))
                for key, val in w2cm.items():
                   w.writerow([key, val])
                showrealiability(userCredit)
                """
                accuracy = 0 #lfc.getaccuracy(truthfile, e2lpd, label_set)
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, accuracy, 'lfc', n_iter]
        elif method == 'kos':
                print('assessing KOS method')
                workermodel = 'cm'
                #result_dst =  os.path.join(os.getcwd(), 'methods\\l_KOS\\kos_result.csv')                               

                #kosInfer= kos.KOSInfer(datafile, truthfile, result_dst )
                #e2lpd = kosInfer.run()
                label_set = ['0', '1']
                #os.chdir(os.path.dirname(__file__))
                #os.chdir('c:/users/tahma/documents/farnazii/research/trust/code/truthinference_shared/attack_strategy/')                
                accuracy = 0.0 #kosInfer.getaccuracy(truthfile, e2lpd, label_set)
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, 0, 'kos', n_iter]

        elif method == 'bcc':
                print('assessing BCC method')
                workermodel = 'cm'
                cf =  correctPath_os(os.path.join(os.getcwd(), 'methods\\bcc\\data\\CF.csv'))
                ans_dst =  correctPath_os(os.path.join(os.getcwd(), 'methods\\bcc\\data\\answers.csv'))                              
                shutil.copy(datafile, ans_dst)
                
                src =  correctPath_os(os.path.join(os.getcwd(), 'methods\\bcc\\Results\\endpoints.csv') )
                #dst = os.path.join(os.getcwd(), 'methods\\bcc\\Results\\All\\endpoints.csv') 
                #shutil.move(src, dst)
                #end = dst
                
                mybcc= bcc.BCCInfer(ans_dst, cf, src )
                e2lpd = mybcc.run()
                label_set = ['0', '1']
                os.chdir(os.path.dirname(__file__))
                #os.chdir('c:/users/tahma/documents/farnazii/research/trust/code/truthinference_shared/attack_strategy/')
                #truthfile = os.path.join(os.getcwd(), 'data\\truth.csv')
                accuracy = mybcc.getaccuracy(truthfile, e2lpd, label_set)
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, accuracy, 'bcc', n_iter]
        elif method == 'zc':
                print('assessing ZC method')
                workermodel = 'cm'
                #e2wl,w2el,label_set = zc.gete2wlandw2el(datafile)
                #e2lpd, wm= zc.EM(e2wl,w2el,label_set).Run()

                #userCredit = getRelability(w2cm)
                
                #credfile = dirname +'\\'+ subdirectory+'\\zc_credibility{}.csv'.format(num_adv)
                #w = csv.writer(open(credfile, "w"))
                #for key, val in w2cm.items():
                #  w.writerow([key, val])
                #showrealiability(userCredit)
                
                accuracy = 0 # zc.getaccuracy(truthfile, e2lpd, label_set)
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, accuracy, 'zc', n_iter]

                #print('Hey you! work better!')
    return change_adver_accu


def getRelability(w2cm):
    reliability = {}
    for k, v in w2cm.items():
        credit = {}
        credit['0'] = v['0']['0']
        credit['1'] = v['1']['1']
        #credit = float("{0:.5f}".format(credit))
        reliability[k] = credit 
    return reliability


def showrealiability(mydict):
    
    xs = []
    ys = []
    for v in mydict.values():
        xs.append(v['0'])
        ys.append(v['1'])
    #xs,ys = zip(*mydict.values())
    labels = mydict.keys()   

    # display
    plt.figure(figsize=(10,8))
    plt.title('Reliability Plot', fontsize=20)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.scatter(xs, ys, marker = 'o')
    for label, x, y in zip(labels, xs, ys):
        plt.annotate(label, xy = (x, y))
    plt.show()
    
#real data 
def recognizibility_partial(num_adv, datafile, n_iter, truthfile, dirname , subdirectory, nworkers, istargeted, target_list, knowledge_rate):
    
    change_adver_accu = pd.DataFrame(columns = ['num_attackers', 'accuracy', 'method', 'iter'])
    rate_change_adv = pd.DataFrame(columns = ['num_attackers', 'accuracy', 'method', 'iter'])

    #models = ['mv','em','mvsoft','mvhard', 'bcc', 'lfc']#, 'kos','zen']
    models = ['em','bcc']#, 'kos','zen']

    for method in models:
        if method == 'em':
                print('assessing EM method')
                workermodel = 'cm'
                e2wl,w2el,label_set = em.gete2wlandw2el(datafile) # generate structures to pass into EM
                iterations = 10 # EM iteration number
                initquality = 0.7
                e2lpd, w2cm = em.EM(e2wl,w2el,label_set,initquality,workermodel, datafile).Run(iterations)
                
                
                #print (w2cm)    
                userCredit = getRelability(w2cm)
                
                credfile = correctPath_os( os.path.join(dirname , subdirectory,'kn_em_credibility{}_kn{}.csv'.format(num_adv, knowledge_rate)))
                w = csv.writer(open(credfile, "w"))
                for key, val in w2cm.items():
                   w.writerow([key, val])
                showrealiability(userCredit)
                
                #remain = '\\em_a{}_b{}_aa{}_ab{}_p{}.csv'.format(average_cm[0], average_cm[1],average_cm[2], average_cm[3],num_adv) 
                #save_user_reliability_path = os.path.join(os.getcwd(),graphDir + remain)
                #w = csv.writer(open(save_user_reliability_path, "w"))
                #for key, val in w2cm.items():
                #   w.writerow([key, val])
                
                #print (e2lpd)
#                TP, FP, TN, FN = perf_measure(true_labels, ah_zhat)

                #truthfile =  os.path.join(os.getcwd(), 'data\\truth.csv')
                if istargeted:
                     accuracy = em.targetedSuccess_rate(target_list,truthfile, e2lpd, label_set)
                     rate_change_adv.loc[change_adver_accu.shape[0]] = [num_adv, accuracy , 'em', n_iter]

                accuracy = em.getaccuracy(truthfile, e2lpd, label_set)
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, accuracy, 'em', n_iter]
#                rates.loc[rates.shape[0]] = [num_adv, 'em', n_iter, 0,0,0,0]   
        elif method == 'bcc':
                print('assessing BCC method')
                workermodel = 'cm'
                cf = correctPath_os( os.path.join(os.getcwd(), 'methods\\bcc\\data\\CF.csv'))
                ans_dst =  correctPath_os(os.path.join(os.getcwd(), 'methods\\bcc\\data\\answers.csv') )                              
                shutil.copy(datafile, ans_dst)
                
                src =  correctPath_os(os.path.join(os.getcwd(), 'methods\\bcc\\Results\\endpoints.csv') )
                credibility_file = correctPath_os(os.path.join(os.getcwd(), 'methods\\bcc\\Results\\credibility.csv') )
                #dst = os.path.join(os.getcwd(), 'methods\\bcc\\Results\\All\\endpoints.csv') 
                #shutil.move(src, dst)
                #end = dst
                
                mybcc= bcc.BCCInfer(ans_dst, cf, src )
                e2lpd = mybcc.run()
                
                
                w2cm = creatw2cm(credibility_file)
                
                userCredit = getRelability(w2cm)
                
                credfile = correctPath_os( os.path.join(dirname , subdirectory,'kn_bcc_credibility{}_kn{}.csv'.format(num_adv, knowledge_rate)))
                w = csv.writer(open(credfile, "w"))
                for key, val in w2cm.items():
                   w.writerow([key, val])
                showrealiability(userCredit)
                
                
                
                label_set = ['0', '1']
                
                os.chdir(os.path.dirname(__file__))
                #os.chdir('c:/users/tahma/documents/farnazii/research/trust/code/truthinference_shared/attack_strategy/')
                #truthfile = os.path.join(os.getcwd(), 'data\\truth.csv')
                if istargeted:
                    accuracy = mybcc.targetedSuccess_rate(target_list,truthfile, e2lpd, label_set)
                    rate_change_adv.loc[change_adver_accu.shape[0]] = [num_adv, accuracy , 'bcc', n_iter]
                accuracy = mybcc.getaccuracy(truthfile, e2lpd, label_set)
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, accuracy, 'bcc', n_iter]
            #elif method == 'kos':
                #print('Hey you! work better!')
    return change_adver_accu , rate_change_adv  


# For RealData
def InferTruth_runAllMethods(num_adv, datafile, n_iter, truthfile, dirname , subdirectory, nworkers, istargeted, target_list):
    change_adver_accu = pd.DataFrame(columns = ['num_attackers', 'accuracy', 'method', 'iter'])
    rate_change_adv = pd.DataFrame(columns = ['num_attackers', 'accuracy', 'method', 'iter'])
    accuracy_dict = {'mv':0, 'em':0, 'bcc':0}
    fscore = {'mv':0, 'em':0, 'bcc':0}
    #models = ['mv','em','mvsoft','mvhard', 'bcc', 'lfc']#, 'kos','zen']
    models = ['mv','em','mvsoft','mvhard', 'bcc', 'lfc', 'kos','zc']#, 'kos','zen']

    for method in models:
        if method == 'mv':
                 print('assessing MV method')
                 e2wl,w2el,label_set = OMV.gete2wlandw2el(datafile)
                 e2lpd = OMV.MV(e2wl,w2el,label_set).Run()
                 if istargeted:
                     mvacc = OMV.targetedSuccess_rate(target_list,truthfile, e2lpd, label_set)
                     rate_change_adv.loc[change_adver_accu.shape[0]] = [num_adv, mvacc , 'mv', n_iter]
                     
                 mvacc = OMV.getaccuracy(truthfile, e2lpd, label_set)                  
                 change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, mvacc , 'mv', n_iter]
                 acc,_,_, fscr = OMV.scores(truthfile, e2lpd, label_set)                  
                 accuracy_dict['mv']= acc
                 fscore['mv'] = fscr
        elif method == 'mvsoft':
                print('assessing MVSOFT method')
                """
                e2wl,w2el,label_set = mvsoft.gete2wlandw2el(datafile)
                firstacc, bestacc, index, removedworkers = mvsoft.MVSoft(e2wl,w2el,label_set, truthfile).Run()
                if index != 0:
                    credfile =correctPath_os( dirname +'\\'+ subdirectory+'\\mvsoft_credibility{}.csv'.format(num_adv))
                    with open(credfile, "w") as output:
                        writer = csv.writer(output, lineterminator='\n')
                        for val in removedworkers:
                            writer.writerow([val])  
                            
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, bestacc , 'mvsoft', n_iter]
                """
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, 0 , 'mvsoft', n_iter]
                
        elif method == 'mvhard':
                #e2wl,w2el,label_set = mvhard.gete2wlandw2el(datafile)
                #firstacc, bestacc, index = mvhard.MVHard(e2wl,w2el,label_set, truthfile).Run()
                bestacc = 0.0
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, bestacc , 'mvhard', n_iter]

        elif method == 'em':
                print('assessing EM method')
                workermodel = 'cm'
                e2wl,w2el,label_set = em.gete2wlandw2el(datafile) # generate structures to pass into EM
                iterations = 20 # EM iteration number
                initquality = 0.7
                e2lpd, w2cm = em.EM(e2wl,w2el,label_set,initquality,workermodel, datafile).Run(iterations)
                
                
                
                
                #print (w2cm)    
                userCredit = getRelability(w2cm)
                
                credfile = correctPath_os( os.path.join(dirname , subdirectory,'em_credibility{}.csv'.format(num_adv)))
                w = csv.writer(open(credfile, "w"))
                for key, val in w2cm.items():
                   w.writerow([key, val])
                showrealiability(userCredit)
                
                #remain = '\\em_a{}_b{}_aa{}_ab{}_p{}.csv'.format(average_cm[0], average_cm[1],average_cm[2], average_cm[3],num_adv) 
                #save_user_reliability_path = os.path.join(os.getcwd(),graphDir + remain)
                #w = csv.writer(open(save_user_reliability_path, "w"))
                #for key, val in w2cm.items():
                #   w.writerow([key, val])
                
                #print (e2lpd)
#                TP, FP, TN, FN = perf_measure(true_labels, ah_zhat)

                #truthfile =  os.path.join(os.getcwd(), 'data\\truth.csv')
                if istargeted:
                     accuracy = em.targetedSuccess_rate(target_list,truthfile, e2lpd, label_set)
                     rate_change_adv.loc[change_adver_accu.shape[0]] = [num_adv, accuracy , 'em', n_iter]

                #accuracy = em.getaccuracy(truthfile, e2lpd, label_set)
                accuracy,_,_,fscr = em.scores(truthfile, e2lpd, label_set)
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, accuracy, 'em', n_iter]
                accuracy_dict['em']= acc
                fscore['em'] = fscr

#                rates.loc[rates.shape[0]] = [num_adv, 'em', n_iter, 0,0,0,0]   
        elif method == 'zc':
                print('assessing ZC method')
                """
                workermodel = 'cm'
                e2wl,w2el,label_set = zc.gete2wlandw2el(datafile)
                e2lpd, wm= zc.EM(e2wl,w2el,label_set).Run()

                userCredit = getRelability(w2cm)
                
                credfile = dirname +'\\'+ subdirectory+'\\zc_credibility{}.csv'.format(num_adv)
                w = csv.writer(open(credfile, "w"))
                for key, val in w2cm.items():
                   w.writerow([key, val])
                showrealiability(userCredit)
                """
                accuracy = 0 #zc.getaccuracy(truthfile, e2lpd, label_set)
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, accuracy, 'zc', n_iter]

        elif method == 'lfc':
                print('assessing LFC method')
                """
                workermodel = 'cm'
                e2wl,w2el,label_set = lfc.gete2wlandw2el(datafile) # generate structures to pass into EM
                iterations = 20 # EM iteration number
                initquality = 0.7
                e2lpd, w2cm = lfc.EM(e2wl,w2el,label_set,initquality,workermodel, datafile).Run(iterations)
                #print (w2cm)    
                userCredit = getRelability(w2cm)
                
                credfile = dirname +'\\'+ subdirectory+'\\LFC_credibility{}.csv'.format(num_adv)
                w = csv.writer(open(credfile, "w"))
                for key, val in w2cm.items():
                   w.writerow([key, val])
                showrealiability(userCredit)
                """
                accuracy = 0 #lfc.getaccuracy(truthfile, e2lpd, label_set)
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, accuracy, 'lfc', n_iter]
        elif method == 'kos':
                print('assessing KOS method')
                """
                workermodel = 'cm'
                res_name = "accuracy_{}.txt".format(nworkers)
                result_dst = os.path.join(os.getcwd(), 'methods\\l_KOS\\'+res_name)                               

                kosInfer= kos.KOSInfer(datafile, truthfile, result_dst )
                """
                accuracy = 0 # kosInfer.run()
                #label_set = ['0', '1']
                #accuracy = kosInfer.getaccuracy(e2lpd, label_set)
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, accuracy, 'kos', n_iter]

        elif method == 'bcc':
                print('assessing BCC method')
                workermodel = 'cm'
                cf = correctPath_os( os.path.join(os.getcwd(), 'methods\\bcc\\data\\CF.csv'))
                ans_dst =  correctPath_os(os.path.join(os.getcwd(), 'methods\\bcc\\data\\answers.csv') )                              
                shutil.copy(datafile, ans_dst)
                
                src =  correctPath_os(os.path.join(os.getcwd(), 'methods\\bcc\\Results\\endpoints.csv') )
                credibility_file = correctPath_os(os.path.join(os.getcwd(), 'methods\\bcc\\Results\\credibility.csv') )
                #dst = os.path.join(os.getcwd(), 'methods\\bcc\\Results\\All\\endpoints.csv') 
                #shutil.move(src, dst)
                #end = dst
                
                mybcc= bcc.BCCInfer(ans_dst, cf, src )
                e2lpd = mybcc.run()
                
                
                w2cm = creatw2cm(credibility_file)
                
                userCredit = getRelability(w2cm)
                
                credfile = correctPath_os( os.path.join(dirname , subdirectory,'bcc_credibility{}.csv'.format(num_adv)))
                w = csv.writer(open(credfile, "w"))
                for key, val in w2cm.items():
                   w.writerow([key, val])
                showrealiability(userCredit)
                
                
                
                label_set = ['0', '1']
                
                os.chdir(os.path.dirname(__file__))
                #os.chdir('c:/users/tahma/documents/farnazii/research/trust/code/truthinference_shared/attack_strategy/')
                #truthfile = os.path.join(os.getcwd(), 'data\\truth.csv')
                if istargeted:
                    accuracy = mybcc.targetedSuccess_rate(target_list,truthfile, e2lpd, label_set)
                    rate_change_adv.loc[change_adver_accu.shape[0]] = [num_adv, accuracy , 'bcc', n_iter]
                accuracy = mybcc.getaccuracy(truthfile, e2lpd, label_set)
                acc,_,_,fscr = mybcc.scores(truthfile, e2lpd, label_set)
                
                change_adver_accu.loc[change_adver_accu.shape[0]] = [num_adv, accuracy, 'bcc', n_iter]
                accuracy_dict['bcc']= acc
                fscore['bcc'] = fscr
            #elif method == 'kos':
                #print('Hey you! work better!')
    return change_adver_accu , rate_change_adv, accuracy_dict, fscore
  
                
def getAccuracyofHonestUser(datafile,truthfile):
     
#    models = ['mv','em']#, 'kos','zen']
#    for method in models:
        #if method == 'mv':
                 #ah_zhat = MV.predict_truth(attack_df, threshold)
                 #ah_zhat = ah_zhat.astype(int)
                 #wronglyPredicted = np.logical_xor(true_labels, ah_zhat)
                 #list_wrong_predicted = np.where( wronglyPredicted ==True )
                 #error_rate = len(list_wrong_predicted[0])/ len(true_labels) # ghalat predict shode             
                 #acc[method] = mv.getAccuracy()
                 e2wl,w2el,label_set = OMV.gete2wlandw2el(datafile)
                 e2lpd = OMV.MV(e2wl,w2el,label_set).Run()
                 mvacc = OMV.getaccuracy(truthfile, e2lpd, label_set)                  
                 return mvacc
    
