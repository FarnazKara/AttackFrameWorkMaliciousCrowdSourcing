# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:33:32 2017

@author: tahma
"""
import numpy as np

def predict():
    return 

def getAccuracy():
    return 0

def get_Percision_recall():
    return 9
    

def predict_truth(worker_tasks_label, treshhold):
          Reports = worker_tasks_label#.iloc[1:,1:]
          nlocs = Reports.shape[1]
          nusers = Reports.shape[0]
          hat = np.zeros(nlocs)
          
          for j in range(nlocs):
              p = Reports.iloc[:,j].tolist().count(1)/nusers
              p = '%.6f'%p
              p = float(p)
              print(treshhold)
              if p >= treshhold: # Reports.iloc[:,j].tolist().count(0):
                  hat[j] = 1
              else:
                  hat[j]= 0
          return hat

