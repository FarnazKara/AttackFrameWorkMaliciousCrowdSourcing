import math
import csv
import random
import sys
from methods.c_MV import method as OMV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class EM:
    def __init__(self,e2wl,w2el,label_set, initquality, workermodel, datafile):
        self.e2wl = e2wl
        self.w2el = w2el
        self.wmodel = workermodel
        self.workers = self.w2el.keys()
        self.label_set = label_set
        if workermodel == 'prob':
            self.initalquality = initquality
        else:
            #initialize by MV method 
            self.initalquality = OMV.initialize_em(datafile)
             


# E-step
    def Update_e2lpd(self):
        self.e2lpd = {}

        for example, worker_label_set in self.e2wl.items():
            lpd = {}
            total_weight = 0

            for tlabel, prob in self.l2pd.items():
                weight = prob
                for (w, label) in worker_label_set:
                    weight *= self.w2cm[w][tlabel][label]
                
                lpd[tlabel] = weight
                total_weight += weight
                

            for tlabel in lpd:
                if total_weight == 0:
                    # uniform distribution 
                    lpd[tlabel] = 1.0/len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel]*1.0/total_weight
            
            self.e2lpd[example] = lpd



#M-step

    def Update_l2pd(self):
        for label in self.l2pd:
            self.l2pd[label] = 0

        for _, lpd in self.e2lpd.items():
            for label in lpd:
                self.l2pd[label] += lpd[label]

        for label in self.l2pd:
            self.l2pd[label] *= 1.0/len(self.e2lpd)


            
    def Update_w2cm(self):

        for w in self.workers:
            for tlabel in self.label_set:
                for label in self.label_set:
                    self.w2cm[w][tlabel][label] = 0


        w2lweights = {}
        for w in self.w2el:
            w2lweights[w] = {}
            for label in self.label_set:
                w2lweights[w][label] = 0
            for example, _ in self.w2el[w]:
                for label in self.label_set:
                    w2lweights[w][label] += self.e2lpd[example][label]

            
            for tlabel in self.label_set:
                # TODO: why it sets to initialquality????
                if w2lweights[w][tlabel] == 0:
                    for label in self.label_set:
                        if tlabel == label:
                            if self.wmodel =='prob':
                                self.w2cm[w][tlabel][label] = self.initalquality
                            else:
                                self.w2cm[w][tlabel][label] = self.initalquality[w]
                        else:
                            if self.wmodel =='prob':
                                self.w2cm[w][tlabel][label] = (1-self.initalquality)*1.0/(len(self.label_set)-1)
                            else:
                                self.w2cm[w][tlabel][label] = (1-self.initalquality[w])*1.0/(len(self.label_set)-1)

                    continue

                for example, label in self.w2el[w]:
                    
                        self.w2cm[w][tlabel][label] += self.e2lpd[example][tlabel]*1.0/w2lweights[w][tlabel]



        return self.w2cm
                    

    
    

                     

#initialization
    def Init_l2pd(self):
        #uniform probability distribution
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0/len(self.label_set)
        return l2pd

    def Init_w2cm(self):
        w2cm = {}
        if self.wmodel == 'prob':
            for worker in self.workers:
                w2cm[worker] = {}
                for tlabel in self.label_set:
                    w2cm[worker][tlabel] = {}
                    for label in self.label_set:
                        if tlabel == label:
                            w2cm[worker][tlabel][label] = self.initalquality
                        else:
                            w2cm[worker][tlabel][label] = (1-self.initalquality)/(len(self.label_set)-1)
        else:
            for worker in self.workers:
                w2cm[worker] = {}
                for tlabel in self.label_set:
                    w2cm[worker][tlabel] = {}
                    for label in self.label_set:
                        if tlabel == label:
                            w2cm[worker][tlabel][label] = self.initalquality[worker]
                        else:
                            w2cm[worker][tlabel][label] = (1-self.initalquality[worker])/(len(self.label_set)-1)


        return w2cm

    def Run(self, iterr = 20):
        
        self.l2pd = self.Init_l2pd()
        self.w2cm = self.Init_w2cm()

        while iterr > 0:
            # E-step
            self.Update_e2lpd() 

            # M-step
            self.Update_l2pd()
            self.Update_w2cm()

            # compute the likelihood
            # print self.computelikelihood()

            iterr -= 1
        
        return self.e2lpd, self.w2cm


    def computelikelihood(self):
        
        lh = 0

        for _, worker_label_set in self.e2wl.items():
            temp = 0
            for tlabel, prior in self.l2pd.items():
                inner = prior
                for worker, label in worker_label_set:
                    inner *= self.w2cm[worker][tlabel][label]
                temp += inner
            
            lh += math.log(temp)
        
        return lh


###################################
# The above is the EM method (a class)
# The following are several external functions 
###################################

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
            
#            y_truth.append(int(e2truth[e].split('.')[0]))
#            y_pred.append(int(truth.split('.')[0]))
            
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
        print("fscore , acc ", f1_score(y_truth, y_pred), accuracy_score(y_truth, y_pred))
        return tcount*1.0/count, f1_score(y_truth, y_pred), recall_score(y_truth, y_pred), precision_score(y_truth, y_pred),accuracy_score(y_truth, y_pred)



#IChange it
def gete2wlandw2el(datafile):
    e2wl = {}
    w2el = {}
    label_set=[]
    
    f = open(datafile, 'r')
    reader = csv.reader(f)
    data = [r for r in reader]

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
    


if __name__ == "__main__":
    

    datafile = sys.argv[1]
    e2wl,w2el,label_set = gete2wlandw2el(datafile) # generate structures to pass into EM
    iterations = 20 # EM iteration number
    initquality = 0.7
    e2lpd, w2cm = EM(e2wl,w2el,label_set,initquality).Run(iterations)

    print (w2cm)
    print (e2lpd)

    # truthfile = r'./truth.csv'
    # accuracy = getaccuracy(truthfile, e2lpd, label_set)
    # print accuracy
    

