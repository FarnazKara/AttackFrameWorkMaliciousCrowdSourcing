import csv, random
import subprocess
import os
import pandas as pd
sep = ","
exec_cs = True

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class BCCInfer:

    
    def __init__(self, ansfile, cfpath, endpath ):
        self.answer_filename =  ansfile #os.path.join(os.getcwd(), 'data\\answers.csv')
        self.cf_file = cfpath
        self.end_file = endpath
        #print(cfpath)
        
    def run(self):
        
        answer_list = []
    
         
        """
        myf = open(self.answer_filename, 'r+')    # pass an appropriate path of the required file
        lines = myf.readlines()
        myf.seek(0) 
        myf.write(lines[0])
        # n is the line number you want to edit; subtract 1 as indexing of list starts from 0
        myf.close()   # close the file and reopen in write mode to enable writing to file; you can also open in append mode and use "seek", but you will have some unwanted old data if the new data is shorter in length.
        """
        
            
        with open(self.answer_filename) as f:
            f.readline()
            
            for line in f:
                if not line:
                    continue
                parts = line.strip().split(sep)
                item_name, worker_name, worker_label = parts[:3]
                answer_list.append([worker_name, item_name, worker_label])
    
        #datafile =  os.path.join(os.getcwd(), 'data\\CF.csv')
    
        with open(self.cf_file, "w") as f:
            for piece in answer_list:
                f.write(",".join(piece))
                f.write("\n")

                
        #df = pd.read_csv(self.answer_filename, sep=',', header=None)
        #df[1:].to_csv(self.cf_file, sep=',', encoding='utf-8',header=None,index=False)
        os.chdir(os.path.dirname(__file__))
        
        if exec_cs:
            #commands.getoutput("/bin/rm Results/endpoints.csv")
            subprocess.call("C:/Windows/system32/cmd.exe", shell=True)
            subprocess.getoutput("CommunityBCCSourceCode.exe")
        if os.path.isfile(self.end_file) == False:
            subprocess.call("C:/Windows/system32/cmd.exe", shell=True)
            subprocess.getoutput("CommunityBCCSourceCode.exe")
            
            
        e2lpd = {}
        
        
        #datafile =  os.path.join(os.getcwd(), 'methods\\f_bcc\\Results\\endpoints.csv')
        with open(self.end_file) as f:
            for line in f:
                parts = line.strip().split(sep)
                e2lpd[parts[0]] = {}
                for i, v in enumerate(parts[1:]):
                    e2lpd[parts[0]][str(i)] = float(v)
    
        return (e2lpd)
    
 
    def targetedSuccess_rate(self, targeted_list, truthfile, e2lpd, label_set):
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
                
#            y_truth.append(int(e2truth[e].split('.')[0]))
#            y_pred.append(int(truth.split('.')[0]))
            if truth.split('.')[0] == e2truth[e].split('.')[0]:
                tcount += 1
    
        return tcount*1.0/count, f1_score(y_truth, y_pred), recall_score(y_truth, y_pred), precision_score(y_truth, y_pred),accuracy_score(y_truth, y_pred)

    
if __name__ == "__main__":
    datafile = os.path.join(os.getcwd(), 'data\\answers.csv')

    
    cf =  os.path.join(os.getcwd(), 'data\\CF.csv')
    end =  os.path.join(os.getcwd(), 'Results\\endpoints.csv') 
    truthfile =  os.path.join(os.getcwd(), 'data\\truth.csv')
    bcctest = BCCInfer(datafile, cf, end)
    e2lpd = bcctest.run()   
    label_set = ['0', '1']
    accuracy = bcctest.getaccuracy(truthfile, e2lpd, label_set)
    print(accuracy)

