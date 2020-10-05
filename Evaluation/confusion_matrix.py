# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:18:16 2017

@author: tahma
"""
from scipy.stats import beta
import scipy.stats as stats
from statistics import mean
from collections import Counter
import numpy as np
import random


class Confusion_Matrix(object):

    def __init__(self):
        self.TT = []
        self.TF = []
        self.FT = []
        self.FF = []
        
        self.TTcdf = []
        self.TFcdf = []
        self.FTcdf = []
        self.FFcdf = []
            
    def setCM(self,cm): #array 
       
        s = len(cm.FFcdf)
        self.TTcdf = cm.TTcdf
        self.TFcdf = [1-cm.TTcdf[i] for i in range(s)]
        self.FTcdf = [1-cm.FFcdf[i] for i in range(s)]
        self.FFcdf = cm.FFcdf
        self.TF = self.TFcdf
        self.FT = self.FTcdf
        self.FF = self.FFcdf 
        
    def getBetaParams(self,mean_dist, svd):
        if svd != 0:
            a = -1.0 * mean_dist *(svd* svd + mean_dist * mean_dist - mean_dist) / (svd*svd)
            b = (mean_dist-1)* (svd* svd + mean_dist * mean_dist - mean_dist) / (svd*svd)
            return a, b
        else: 
            return -1, -1
    
    def createCM(self, users, mean_alpha, svd_alpha, mean_beta, svd_beta):
        
        if mean_alpha == 1.0: 
            rnd_alpha = [1] * users
            self.TT = rnd_alpha
            self.TTcdf = self.TT
            self.TF = [abs(1 - x) for x in self.TT]
            self.TFcdf = self.TF
        if mean_alpha == 0.0: 
            rnd_alpha = [0] * users
            self.TT = rnd_alpha
            self.TTcdf = self.TT
            
            
            self.TF = [abs(1 - x) for x in self.TT]
            self.TFcdf = self.TF
        if mean_beta == 1.0:
            rnd_beta = [1] * users
            self.FF = rnd_alpha
            
            self.FFcdf = self.FF
            self.FT = [abs(1 - x) for x in self.FF]
            self.FTcdf = self.FT
            
        if mean_beta == 0.0: 
            rnd_alpha = [0] * users
            self.FF = rnd_alpha
            self.FFcdf  = self.FF 
            
            
            self.FT = [abs(1 - x) for x in self.FF]
            self.FTcdf = self.FT
        
        if len(self.TT) == 0:
            q = np.random.rand(1000)
            alpha_values, beta_values = self.getBetaParams(mean_alpha, svd_alpha)
            if alpha_values == -1 and beta_values == -1: 
                rnd_alpha = [mean_alpha] * users
                self.TT = rnd_alpha
                self.TTcdf  = self.TT 
                self.TF = [abs(1 - x) for x in self.TT]
                self.TFcdf = self.TF
            else:
                vals = beta.ppf(q, alpha_values, beta_values)
                rnd_alpha = random.sample(list(vals), users)
                distalpha = beta(alpha_values, beta_values)
                self.TT = rnd_alpha
                self.TF = [abs(1 - x) for x in self.TT]

        if len(self.FF) == 0:
        
            q=np.random.rand(1000)
            alpha_values, beta_values = self.getBetaParams(mean_beta, svd_beta)
            if alpha_values == -1 and beta_values == -1:
                rnd_alpha = [mean_beta] * users
                self.FF = rnd_alpha
                self.FFcdf  = self.FF 
                self.FT = [abs(1 - x) for x in self.FF]
                self.FTcdf = self.FT
            else:
                vals = beta.ppf(q, alpha_values, beta_values )
                rnd_beta = random.sample(list(vals), users)
                distbeta = beta(alpha_values, beta_values)
                self.FF = rnd_beta
                self.FT = [abs(1 - x) for x in self.FF]
        
        self.FTcdf = self.FT
        self.FFcdf  = self.FF 
        self.TTcdf = self.TT
        self.TFcdf = self.TF
        #return distalpha, distbeta
        
        
    def getReliability(self, predicted, truth):
        xorset = [predicted[i] ^ truth[i] for i in range(len(predicted))] 
        self.TT = xorset.count(1) / len(predicted)
        self.TF = 1- self.TT
        self.FF = xorset.count(0) / len(predicted)
        self.FT = 1- self.FF
        
    
    
    
    
