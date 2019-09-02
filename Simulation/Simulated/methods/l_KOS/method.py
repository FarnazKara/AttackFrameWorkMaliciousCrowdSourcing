__author__ = 'JasonLee'

import subprocess
import sys
import os, csv, random
import math, shutil

sep = ","
exec_cs = True

class KOSInfer:    
    def __init__(self, ansfile, truthfile, acc_res_file ):
        self.answer_filename =  ansfile #os.path.join(os.getcwd(), 'data\\answers.csv')
        self.truth_file = truthfile
        self.result = acc_res_file
        #print(cfpath)

    def run(self):  
        
        ans_dst = os.path.dirname(__file__)                        
        ans_dst_name = ans_dst + "\\answers.csv"
        shutil.copy(self.truth_file, ans_dst)
        shutil.copy(self.answer_filename, ans_dst_name)

        #subprocess.getoutput("matlab -nojvm -nodisplay -nosplash -r " + "\"" + "filename = '" +
        #           filepath + "'; " + "prepare\" -logfile log")

        #subprocess.getoutput("matlab -nojvm -nodisplay -nosplash -wait -r " +  "prepare\" -logfile log")

    
        
        subprocess.call('matlab -nosplash -wait -r "prepare"', cwd =os.path.dirname(__file__),  shell=0)



        """
        e2lpd = {}
        with open(self.result_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if math.isnan(float(parts[1])):
                    e2lpd[parts[0]] = {'0': 0, '1': 0}
                else:
                    e2lpd[parts[0]] = {'0': 0, '1': float(parts[1])}
        
        """
        
        file = open(self.result,'r')
        accuracy = float(file.readline())
        file.close()
        return accuracy
    
    
