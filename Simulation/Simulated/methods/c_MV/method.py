import copy
import random
import sys
import csv
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class MV:

    def __init__(self,e2wl,w2el,label_set):

        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = self.w2el.keys()
        self.label_set = label_set


    def Run(self):

        e2wl = self.e2wl
        e2lpd={}
        for e in e2wl:
            e2lpd[e]={}

            # multi label
            for label in self.label_set:
                e2lpd[e][label] = 0
            # e2lpd[e]['0']=0
            # e2lpd[e]['1']=0

            for item in e2wl[e]:
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
    

def targetedSuccess_rate(targeted_list, truthfile, e2lpd, label_set):
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
    
            if e not in targeted_list:
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
    
            if truth.split('.')[0] != e2truth[e].split('.')[0]:
                tcount += 1
    
        return tcount*1.0/count



    

def getaccuracy(truthfile, e2lpd, label_set):
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
    
def getaccuracy_fscore(truthfile, e2lpd, label_set):
        e2truth = {}
        f = open(truthfile, 'r')
        reader = csv.reader(f)
        next(reader)
    
        for line in reader:
            example, truth = line
            e2truth[example] = truth
    
        tcount = 0
        count = 0
        
        y_truth = []
        y_pred = []
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
        print ("my test", tcount*1.0/count, f1_score(y_truth, y_pred), recall_score(y_truth, y_pred), precision_score(y_truth, y_pred),accuracy_score(y_truth, y_pred))
        return tcount*1.0/count, f1_score(y_truth, y_pred), recall_score(y_truth, y_pred), precision_score(y_truth, y_pred),accuracy_score(y_truth, y_pred)


def gete2wlandw2el(datafile):
    e2wl = {}
    w2el = {}
    label_set=[]
    
    f = open(datafile, 'r')
    reader = csv.reader(f)
    data = [r for r in reader]
    #next(reader)
    #nline = 0 

    for line in data:
        if line == []:
            continue
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

#add functions to use in EM 
def getwreliability(w2el,e2lpd):
    cnt = 0
    cntwt = 0
    w2prob = {}
    for key, value in w2el.items():
        cntwt = len(value)
        cnt = 0
        for e,label in value:
            if label == get_tlabel(e2lpd,e): 
                cnt += 1
        if cntwt == 0:
            w2prob[key] = 0
        else:
            w2prob[key] = round(cnt/ cntwt,2)
    return w2prob
    # If she does not do anything, I should assign zzero to that, is it true?      
    
def initialize_em(datafile):
    
    e2wl,w2el,label_set = gete2wlandw2el(datafile)
    e2lpd = MV(e2wl,w2el,label_set).Run()
    wreliability = getwreliability(w2el,e2lpd)
    return wreliability



    
    
if __name__ == "__main__":
    datafile = sys.argv[1]
    e2wl,w2el,label_set = gete2wlandw2el(datafile)
    e2lpd = MV(e2wl,w2el,label_set).Run()

    print(e2lpd)
    

