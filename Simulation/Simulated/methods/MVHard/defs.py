# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:22:56 2018

@author: tahma
"""

class User(object):
    def __init__(self, user_id, is_honest, name=None, avg_rating=None, num_reviews=0, info=None):
        self.id = user_id
        self.parent = None
        self.degM = 0
        self.honest = is_honest
        self.name = name
        self.avg_rating = avg_rating
        self.info = info
        self.num_reviews = num_reviews
        self.alpha = 0.
        self.beta = 0.
        self.reputation = 0.
        self.filtered = False
        self.score = 0

class Object(object):
    def __init__(self, obj_id, truth, name=None, rating=None):
        self.id = obj_id
        self.parent = None
        self.votes = dict()
        self.upvotes = 0
        self.downvotes = 0
        self.truth = truth
        self.conflict = False
        self.name = name
        self.cum_rating = rating
