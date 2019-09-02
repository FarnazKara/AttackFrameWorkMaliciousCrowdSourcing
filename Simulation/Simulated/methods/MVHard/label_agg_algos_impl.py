# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 19:52:55 2018

@author: tahma
"""
from collections import deque, defaultdict
import numpy as np
import sys, os
from sklearn.utils.extmath import logsumexp
import itertools


def accuracy(labels, Objects):
    """
    :param labels: python dict mapping task id to predicted label
    :param Objects: python dict containing task info
    :return:
    """
    
    task_ids = sorted(Objects.keys())
    labels_acc = np.array([labels[obj] == Objects[obj].truth for obj in task_ids], dtype=np.float)
    return labels_acc

# Output simple majority label
def majority_voting(Users, Objects):
    decision = dict()
    for obj in Objects:
        upvotes = Objects[obj].upvotes
        downvotes = Objects[obj].downvotes
        if (upvotes + downvotes <= 0):
            print("err")
            print(obj)

        #assert (upvotes + downvotes > 0)
        if upvotes >= downvotes:
            decision[obj] = 1
        else:
            decision[obj] = -1

    return accuracy(decision, Objects) 

def get_predicted_truth(Users, Objects):
    decision = dict()
    for obj in Objects:
        upvotes = Objects[obj].upvotes
        downvotes = Objects[obj].downvotes
        if (upvotes + downvotes <= 0):
            print("err")
            print(obj)

#        assert (upvotes + downvotes > 0)
        if upvotes >= downvotes:
            decision[obj] = 1
        else:
            decision[obj] = -1

    return decision

# compute hard-penalty reputation score via optimal semi-matching
def compute_rep_scores(B, Users, Objects):
    Q = deque()
    visited = set()
    # M represents the semi-matching where key is the object and value is the user mapped
    M = dict()
    for obj in Objects:
        assert (B.degree(obj) > 0), 'Error in creating conflict graph...'
        # clear the set of visited vertices and the BFS queue
        Q.clear()
        visited.clear()
        # start the new BFS run
        Q.append(obj)
        visited.add(obj)

        best_user = None

        while len(Q) > 0:
            w = Q.popleft()
            test = (w in Objects)
            unmatched = (w not in M)
            if test:
                if unmatched:
                    Neighbors = B.neighbors(w)
                else:
                    Neighbors = [u for u in B.neighbors(w) if u != M[w]]
            else:
                Neighbors = [o for o in B.neighbors(w) if o in M and M[o] == w]

                if best_user is None or Users[w].degM < Users[best_user].degM:
                    best_user = w

            for neigh in Neighbors:
                if neigh not in visited:
                    if test:
                        Users[neigh].parent = w
                    else:
                        Objects[neigh].parent = w
                    Q.append(neigh)
                    visited.add(neigh)

        assert best_user is not None, 'Could not match task:%s to any worker' % obj
        user = best_user
        ohat = Users[user].parent
        M[ohat] = user
        Users[user].degM += 1
        while ohat != obj:
            user = Objects[ohat].parent
            ohat = Users[user].parent
            M[ohat] = user

    return M

# impl. of iterative messaging passing algorithm in Karger et al. 2011 NIPS paper
def KOS_estimates(B, Users, Objects, norm,  num_adv):
    kmax = 100 # max number of iterations of KOS algorithm
    num_users = len(Users)
    num_objects = len(Objects)
    yj2i = np.zeros((num_users, num_objects))
    xi2j = np.zeros((num_objects, num_users))
    A = np.zeros((num_objects, num_users))
    degrees = defaultdict(int)
    odegrees = defaultdict(int)
    for obj in Objects:
        for user in Objects[obj].votes:
            A[obj - num_users][user] = Objects[obj].votes[user]
            if Objects[obj].votes[user] != 0:
                degrees[user] += 1
                odegrees[obj] += 1

    for node in B.nodes():
        if node < num_users:
            continue
        for neigh in B.neighbors(node):
            yj2i[neigh][node - num_users] = np.random.normal(1, 1)

    for _ in range(kmax):
        for obj in Objects:
            xi2j[obj - num_users] = np.dot(A[obj - num_users], yj2i[:, obj - num_users]) - A[obj - num_users] * yj2i[:, obj - num_users]
            if norm:
                xi2j[obj - num_users] /= odegrees[obj]

        for user in Users:
            yj2i[user] = np.dot(A[:, user], xi2j[:, user]) - A[:, user] * xi2j[:, user]
            if norm and degrees[user] != 0:
                yj2i[user] /= degrees[user]

        norm_const = np.abs(yj2i.max())
        assert (norm_const > 0)
        yj2i = yj2i / norm_const

    decision = dict()
    for obj in Objects:
        upvotes = Objects[obj].upvotes
        downvotes = Objects[obj].downvotes
        if (upvotes + downvotes <= 0):
            print("err")
            print(obj)
        assert (upvotes + downvotes > 0)
        val = np.dot(A[obj - num_users], yj2i[:, obj - num_users])
        decision[obj] = np.sign(val)
    
    credfile = os.path.join('kos_credibility{}.csv'.format(5))
    w = csv.writer(open(credfile, "w"))
    #for key, val in w2cm.items():
    #    w.writerow([key, val])
                
    

    return accuracy(decision, Objects)
