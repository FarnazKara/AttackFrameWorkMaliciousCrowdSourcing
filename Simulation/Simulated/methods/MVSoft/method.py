import copy
import random
import sys, os
import csv
from collections import Counter
from statistics import mean
import operator
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class MVSoft:

    def __init__(self,e2wl,w2el,label_set, truthpath):

        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = self.w2el.keys()
        self.label_set = label_set
        self.all_e2lpd = []
        self.truthfile = truthpath


    def getConflictedSet(self):
        tcs = []       
        for e in self.e2wl:
            myset = set()
            for item in self.e2wl[e]:
                myset.add(item[1])
                if len(myset) > 1:
                    tcs.append(e)
                    break
        return tcs
    
    
    def updateworkerTaskList(self, removeWorker):
        for key, val in self.e2wl.items():
            while [removeWorker, '0.0'] in val or [removeWorker, '0'] in val:
                if [removeWorker, '0.0'] in val:
                    val.remove([removeWorker, '0.0'])
                elif [removeWorker, '0'] in val:
                    val.remove([removeWorker, '0'])
                self.e2wl[key] = val
            while [removeWorker, '1'] in val or [removeWorker, '1.0'] in val:
                if [removeWorker, '1'] in val:
                    val.remove([removeWorker, '1']) 
                elif [removeWorker, '1.0'] in val:
                    val.remove([removeWorker, '1.0']) 
                self.e2wl[key] = val
        for key, val in self.e2wl.items():
               if [removeWorker, '0'] in val or [removeWorker, '1'] in val or [removeWorker, '0.0'] in val or [removeWorker, '1.0'] in val:
                   print("Khaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaar\n\n")
        
        del self.w2el[removeWorker]
        self.w2el = self.w2el
        self.workers = self.w2el.keys()

        
        
    def worker_highestPenalty(self,penaltystat):
        selectedworker = max(penaltystat.items(), key=operator.itemgetter(1))[0]
        return selectedworker
        
        
    def Run(self):
        self.all_e2lpd = []
        diff_acc = []
        fscore_acc = []
        e2lpd = self.runMV()
        truthfile = self.truthfile # os.getcwd() + "\\data\\truth.csv"
        
        removedworker = []
        
        self.all_e2lpd.append(e2lpd)
        diff_acc.append(self.getaccuracy(truthfile, e2lpd,self.label_set))
        for i in range(1,10):
            e2wl = self.e2wl
            p2w = {}
            Stw = {}
            workerPenalty={}
            task_conflict_set = self.getConflictedSet()
            for t in task_conflict_set:
                labelspecifictask = []
                labelspecifictask = [e2wl[t][i][1] for i in range(len(e2wl[t]))]
                cntLabels = Counter(labelspecifictask)
                for value in e2wl[t]:
                    w = value[0]
                    #print("in iteration ")
                    #print(i)
                    #print(t)
                    if w in removedworker:
                        print("Error it did not cleaned well"+ w)
                    findans = [list_t[1] for list_t in  self.w2el[w] if list_t[0] == t]
                    if findans == ['1.0'] or findans == ['1']:
                        if findans == ['1.0']:
                            Stw = 1.0/cntLabels['1.0']
                        elif findans == ['1']:
                            Stw = 1.0/cntLabels['1']
                        Stw = float("{0:.3f}".format(Stw))

                        if w in p2w:
                            p2w[w] = p2w[w] + [Stw]
                        else:
                            p2w[w] =  [Stw]
                    elif findans == ['0'] or findans == ['0.0']:
                        if findans == ['0.0']:
                            Stw = 1.0/cntLabels['0.0']
                        elif findans == ['0']:
                            Stw = 1.0/cntLabels['0']
                        Stw = float("{0:.3f}".format(Stw))
                        if w in p2w:
                            p2w[w] = p2w[w] + [Stw]
                        else:
                            p2w[w] = [Stw]
            
            
            for key , value in p2w.items():
                 workerPenalty[key] = float("{0:.3f}".format(mean(value))) 
                 #print(key)
                 #print(workerPenalty[key])
                 
        
            if len(workerPenalty) == 0:
                print("Could not find conflicted tasks")
                if len(diff_acc) == 1:
                    return diff_acc[0], diff_acc[0], 0, removedworker
                else:
                    return diff_acc[0], max(diff_acc), diff_acc.index(max(diff_acc)) , removedworker
            selectedworker = self.worker_highestPenalty(workerPenalty)
            removedworker.append(selectedworker)
            self.updateworkerTaskList(selectedworker)
            e2lpd = self.runMV()
            self.all_e2lpd.append(e2lpd)
            #truthfile = os.getcwd() + "data\\truth.csv"
            diff_acc.append(self.getaccuracy(truthfile, e2lpd,self.label_set))
            _, my_fscore, _,_ = self.getaccuracy_fscore(truthfile, e2lpd,self.label_set)
            fscore_acc.append(my_fscore)
        
        return diff_acc[0], max(diff_acc), diff_acc.index(max(diff_acc)), removedworker, my_fscore
        
    
    def get_e2lpd(self,level):
        return self.all_e2lpd[level]
    
    def runMV(self):
        e2wl = self.e2wl
        e2lpd={}


        for e in e2wl:
            e2lpd[e]={}

            # multi label
            for label in self.label_set:
                e2lpd[e][label] = 0
            # e2lpd[e]['0']=0
            # e2lpd[e]['1']=0
            #print(e)
            for item in e2wl[e]:
                #print(item)
                label=item[1]
                e2lpd[e][label]+= 1

            # alls=e2lpd[e]['0']+e2lpd[e]['1']
            alls = 0
            for label in self.label_set:
                alls += e2lpd[e][label]
            if alls!=0:
                # e2lpd[e]['0']=1.0*e2lpd[e]['0']/alls
                # e2lpd[e]['1']=1.0*e2lpd[e]['1']/alls
                for label in self.label_set:
                    e2lpd[e][label] = 1.0 * e2lpd[e][label] / alls
            else:
                # e2lpd[e]['0']=0.5
                # e2lpd[e]['1']=0.5
                for label in self.label_set:
                    e2lpd[e][label] = 1.0 / len(self.label_set)

        # return self.expand(e2lpd)
        return e2lpd


    def getaccuracy(self,truthfile, e2lpd, label_set):
        e2truth = {}
        f = open(truthfile, 'r')
        reader = csv.reader(f)
        next(reader)
    
        for line in reader:
            example, truth = line
            e2truth[example] = truth
    
        tcount = 0
        count = 0
    
        for e in e2lpd:
    
            if e not in e2truth:
                continue
    
            temp = 0
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]
            
            candidate = []
    
            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)
    
            truth = random.choice(candidate)
    
            count += 1
    
            if truth.split('.')[0] == e2truth[e].split('.')[0]:
                tcount += 1
    
        return tcount*1.0/count
    
    def getaccuracy_fscore(self,truthfile, e2lpd, label_set):
        e2truth = {}
        f = open(truthfile, 'r')
        reader = csv.reader(f)
        next(reader)
    
        for line in reader:
            example, truth = line
            e2truth[example] = truth
    
        y_truth = []
        y_pred = []
        tcount = 0
        count = 0
    
        for e in e2lpd:
    
            if e not in e2truth:
                continue
    
            temp = 0
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]
            
            candidate = []
    
            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)
    
            truth = random.choice(candidate)
    
            count += 1
    
            if (int(e2truth[e].split('.')[0])) == 0:
                y_truth.append(1)
            else:
                y_truth.append(0)
            
            if (int(truth.split('.')[0])) == 0:
                y_pred.append(1)
            else: 
                y_pred.append(0)
            if truth.split('.')[0] == e2truth[e].split('.')[0]:
                tcount += 1
        print ("mysoft test", tcount*1.0/count, f1_score(y_truth, y_pred), recall_score(y_truth, y_pred), precision_score(y_truth, y_pred),accuracy_score(y_truth, y_pred))
        return tcount*1.0/count, f1_score(y_truth, y_pred), recall_score(y_truth, y_pred), precision_score(y_truth, y_pred),accuracy_score(y_truth, y_pred)

        
    
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
    
def get_tlabel(e2lpd, e):
        temp = 0
        cnd_label = -1
        for label in e2lpd[e]:
            if temp < e2lpd[e][label]:
               temp = e2lpd[e][label]
               cnd_label = label    
        return cnd_label



    
if __name__ == "__main__":
    mal = 0.6
    adv_num = 180
    niter = 0
    datafile = os.path.join( os.getcwd() ,'..\\..\\', 'large_data_{}'.format(mal), 'iter_{}_uid_300_tid_12000_dense_5_dout_200'.format(niter),'alpha_1.0_beta_1.0_adv_alpha0.0_adv_beta_0.0_gamma_0', 'union_ans_dens_5_p_{}.csv'.format(adv_num) )# "\\data\\answers.csv"
    datafile = os.path.join( os.getcwd() ,'..\\..\\', 'large_data_{}'.format(mal), 'iter_{}_uid_300_tid_12000_dense_5_dout_200'.format(niter),'alpha_1.0_beta_1.0_adv_alpha0.0_adv_beta_0.0_gamma_0', 'union_ans_dens_5_p_{}.csv'.format(adv_num) )# "\\data\\answers.csv"
    
    e2wl,w2el,label_set = gete2wlandw2el(datafile)
    firstacc, bestacc, index, _, fscore = MVSoft(e2wl,w2el,label_set, truthpath).Run()

    print(firstacc)
    print(bestacc)
    print(index)
    print(fscore)
    

