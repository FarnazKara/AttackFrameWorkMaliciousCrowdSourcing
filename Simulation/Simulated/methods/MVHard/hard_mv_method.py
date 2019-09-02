# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 18:16:51 2018

@author: tahma
"""

import sys
import networkx as nx
from methods.MVHard.defs import User, Object
import numpy as np
import matplotlib.pyplot as plt
import random, os
from heapq import nlargest
from collections import defaultdict, Counter
from scipy.stats import ttest_ind, ttest_rel
import itertools 
from methods.MVHard.label_agg_algos_impl import majority_voting, accuracy, compute_rep_scores, KOS_estimates, get_predicted_truth

fast = False
plt.rc('font', size=15)
plt.rc('ps', useafm=True)
plt.rc('pdf', use14corefonts=True)

np.random.seed(2928)
random.seed(6410)


def penalty_spread_iter(Users, Objects, algo):
    degrees = defaultdict(int)
    for obj in Objects:
        if not Objects[obj].conflict:
            continue
        for user in Objects[obj].votes:
            if Users[user].filtered:
                continue
            if Objects[obj].votes[user] == 1:
                degrees[user] += 1
                # Users[user].reputation += total_votes/float(Objects[obj].upvotes)
                Users[user].reputation += 1 / float(Objects[obj].upvotes)
                # Users[user].reputation += -np.log(float(Objects[obj].upvotes)/total_votes)
            elif Objects[obj].votes[user] == -1:
                # Users[user].reputation += total_votes/float(Objects[obj].downvotes)
                Users[user].reputation += 1 / float(Objects[obj].downvotes)
                # Users[user].reputation += -np.log(float(Objects[obj].downvotes)/total_votes)
                degrees[user] += 1

    if algo == 'soft':
        for user in degrees:
            Users[user].reputation /= degrees[user]


def create_conflict_graph(Objects):
    Conflicts = dict()
    VGraph = nx.Graph()
    for obj in Objects:
        if Objects[obj].conflict:
            obj1 = '+' + str(obj)
            obj2 = '-' + str(obj)
            Conflicts[obj1] = Object(obj1, 1)
            Conflicts[obj2] = Object(obj2, -1)
            VGraph.add_node(obj1)
            VGraph.add_node(obj2)
            for user in Objects[obj].votes:
                if Objects[obj].votes[user] == 1:
                    VGraph.add_edge(user, obj1)
                elif Objects[obj].votes[user] == -1:
                    VGraph.add_edge(user, obj2)

    return VGraph, Conflicts


# compute performance stats of different workers in each dataset
def compute_worker_stats(Users, Objects, B):
    num_uniform = 0
    num_ry_spammers = 0
    for user in Users:
        correct = 0
        alphanum = 0
        alphaden = 0
        betanum = 0
        betaden = 0
        user_responses = Counter()
        for obj in B.neighbors(user):
            vote = Objects[obj].votes[user]
            user_responses.update([vote])
            if vote == Objects[obj].truth:
                correct += 1
                if Objects[obj].truth == 1:
                    alphanum += 1
                    alphaden += 1
                else:
                    betanum += 1
                    betaden += 1
            else:
                if Objects[obj].truth == 1:
                    alphaden += 1
                else:
                    betaden += 1

        Users[user].reliability = float(correct) / B.degree(user)
        Users[user].alpha = float(alphanum) / alphaden if alphaden != 0 else -1
        Users[user].pos_degree = alphaden
        Users[user].neg_degree = betaden
        Users[user].beta = float(betanum) / betaden if betaden != 0 else -1
        # compute spammer score as defined in Raykar-Yu '12
        Users[user].spammer_score = np.abs(Users[user].alpha + Users[user].beta - 1) if Users[user].alpha > -1 and Users[user].beta > -1 else 2
        if len(user_responses) == 1:
            num_uniform += 1
        if Users[user].spammer_score <= 0.05: # 0.05 threshold taken from Raykar-Yu 2012.
            num_ry_spammers += 1

    prevalence = 0.
    # num_not_conflict = 0
    for obj in Objects:
        if Objects[obj].truth == 1:
            prevalence += 1

    print ('#######Printing dataset stats#############')
    print ('Prevalence for +1 tasks is: %.2f' % (prevalence / len(Objects)))
    print ("Number of workers: " + str(len(Users)))
    print ("Number of tasks: " + str(len(Objects)))
    print ("Number of responses: " + str(B.number_of_edges()))
    print ('Number of uniform workers: ' + str(num_uniform))
    print ('Number of raykar-yu spammers: ' + str(num_ry_spammers))



# prepare data of tweets dataset
def tweets_sentiments(dfile, tf):
    #dirfolder = os.getcwd()
    #tf = dirfolder +"\\data\\truth.csv" 
    #dfile = dirfolder+"\\data\\answers.csv" 
    f1 = open(tf, 'r')
    f2 = open(dfile, 'r')

    label_data = f2.readlines()[1:]
    f2.close()
    # num_users = 83
    Users = dict()
    Objects = dict()
    vote_graph = nx.Graph()
    obj_mapping = dict()
    user_degrees = defaultdict(set)

    for line in label_data:
        [oid, uid, vote] = line.rstrip().split(',')
        #[uid, oid, vote] = line.split()
        user_degrees[uid].add(oid)

    udegs = [len(user_degrees[usid]) for usid in user_degrees]
    num_users = len(np.where(np.array(udegs) > 2)[0])
    obj = num_users

    truth_datas = f1.readlines()[1:]
    for line in truth_datas:
        [oid, toid] = line.rstrip().split(',')
        obj_mapping[oid] = obj
        if toid.strip() == '1' or toid.strip() == '1.0' :
            label = 1
        else:
            label = -1
        Objects[obj] = Object(obj, label)
        obj += 1

    user = 0
    user_mapping = dict()
    for line in label_data:
        [oid, uid, vote] = line.rstrip().split(',')
        if oid not in obj_mapping or len(user_degrees[uid]) <= 2:
            continue
        #if uid not in user_mapping:
        if uid not in user_mapping:
            user_mapping[uid] = user
            Users[user] = User(user, True, 1.)
            user += 1

        vobj = obj_mapping[oid]
        worker = user_mapping[uid]
        vote_graph.add_edge(worker, vobj)
        if vote.strip() == '1' or vote.strip() == '1.0' :
            Objects[vobj].votes[worker] = 1
        else:
            Objects[vobj].votes[worker] = -1

    for obj in Objects:
        list_dicts = list(Objects[obj].votes.values())
        dicts_array = np.array(list_dicts) 
        Objects[obj].upvotes = len(np.where(dicts_array == 1)[0])
        Objects[obj].downvotes = len(np.where( dicts_array == -1)[0])
        if Objects[obj].upvotes > 0 and Objects[obj].downvotes > 0:
            Objects[obj].conflict = True
        else:
            Objects[obj].confct = False

    f1.close()
    compute_worker_stats(Users, Objects, vote_graph)
    # print Counter([vote_graph.degree(hitw) for hitw in xrange(num_users)])
    return (Users, Objects, vote_graph)



def real_world_performance(Users, Objects, B, mode):
    """
    :param Users: python dict. mapping worker id to information about each worker
    :param Objects: python dict. mapping task id to information
                   (such as label of each assigned worker, num. of +1/-1 labels, etc.)
                   about the task
    :param B: python networkx bipartite graph representing set of workers assigned for each task
    :param mode: mode of reputation algorithm (soft or hard)
    :return:
    """
    n = len(Users) # total number of workers
    limit = 10 # num. of workers to (iteratively) filter
    flagged = [] # ids of workers who are filtered
    error_rate = defaultdict(list) # dictionary storing error rates of each label agg. algorithm

    # compute error rates of algorithms when considering labels of ALL workers
    
    error_rate['maj'].append(majority_voting(Users, Objects))
    error_rate['iter_norm'].append(KOS_estimates(B, Users, Objects, True, dirname , subdirectory,num_adv))

    """
    error_rate['em'].append(EM_estimates(Users, Objects))
    error_rate['spec_em'].append(spectral_EM_estimates(Users, Objects))
    error_rate['iter'].append(KOS_estimates(B, Users, Objects, False))
    error_rate['iter_norm'].append(KOS_estimates(B, Users, Objects, True))
    """
    # start iteratively removing workers
    for _ in range(1, limit + 1):
        # hard penalty algorithm
        if mode == 'hard':
            # create conflict graph B^cs
            (conflict_graph, conflict_set) = create_conflict_graph(Objects)
            if conflict_set == {}: 
                break
            # compute optimal semi-matching (OSM) on B^cs
            compute_rep_scores(conflict_graph, Users, conflict_set)
            # determine worker with largest degree in OSM
            high = nlargest(1, range(n), key=lambda i: Users[i].degM)[0]
            assert (Users[high].degM > 0)
        # soft penalty algorithm
        else:
            # compute soft-penalty for each worker
            penalty_spread_iter(Users, Objects, mode)
            # compute worker with largest penalty
            high = nlargest(1, range(n), key=lambda i: Users[i].reputation)[0]
            if Users[high].reputation <= 0:
                break
            assert (Users[high].reputation > 0)

        Users[high].filtered = True
        # add worker to filtered list
        flagged.append(high)

        # remove labels of the filtered worker
        for obj in B.neighbors(high):
            if Objects[obj].conflict:
                if Objects[obj].votes[high] > 0:
                    Objects[obj].upvotes -= 1
                else:
                    Objects[obj].downvotes -= 1
                Objects[obj].votes[high] = 0

        # update conflict set of objects
        for obj in Objects:
            if Objects[obj].upvotes == 0 or Objects[obj].downvotes == 0:
                Objects[obj].conflict = False

        # reset penalties and degrees in OSM
        for u in Users:
            Users[u].degM = 0
            Users[u].reputation = 0.

        # compute error rates of algorithms considering labels of only REMAINING workers
        error_rate['pen_maj'].append( majority_voting(Users, Objects))
        """
        error_rate['pen_spec_em'].append(spectral_EM_estimates(Users, Objects))
        error_rate['pen_em'].append(EM_estimates(Users, Objects))
        error_rate['pen_iter'].append(KOS_estimates(B, Users, Objects, False))
        error_rate['pen_iter_norm'].append(KOS_estimates(B, Users, Objects, True))
        """
    for alg in ['maj']:#, 'em', 'iter', 'iter_norm', 'spec_em']:
        base_acc = np.array(error_rate[alg])
        base_percent_acc = np.mean(base_acc, axis=1)
        print ('baseline for %s: %s' % (alg, base_percent_acc))
        with_pen_acc = np.array(error_rate['pen_' + alg])
        if len(with_pen_acc) > 0: 
            with_pen_percent_acc = np.mean(with_pen_acc, axis=1)
            number_of_times_better = np.mean(with_pen_percent_acc >= base_percent_acc)
            best_index = np.argmax(with_pen_percent_acc)
            print ('with %s penalty: %s(removing %d workers)' % (mode, with_pen_percent_acc[best_index], best_index + 1))
            print ('relative t-test p values: %s' % ttest_rel(base_acc, with_pen_acc, axis=1)[1][best_index])
            print ('Filtering workers is better %.3f fraction of the time' % number_of_times_better)
            print ('================')        
    base_acc = np.array(error_rate['iter_norm'])
    base_percent_acc = np.mean(base_acc, axis=1)
    print ('baseline for %s: %s' % ('iter_norm', base_percent_acc))

def part_real_world_performance(Users, Objects, B, mode):
    """
    :param Users: python dict. mapping worker id to information about each worker
    :param Objects: python dict. mapping task id to information
                   (such as label of each assigned worker, num. of +1/-1 labels, etc.)
                   about the task
    :param B: python networkx bipartite graph representing set of workers assigned for each task
    :param mode: mode of reputation algorithm (soft or hard)
    :return:
    """
    n = len(Users) # total number of workers
    limit = 10 # num. of workers to (iteratively) filter
    flagged = [] # ids of workers who are filtered
    error_rate = defaultdict(list) # dictionary storing error rates of each label agg. algorithm

    # compute error rates of algorithms when considering labels of ALL workers
    
    error_rate['maj'].append(majority_voting(Users, Objects))

    """
    error_rate['em'].append(EM_estimates(Users, Objects))
    error_rate['spec_em'].append(spectral_EM_estimates(Users, Objects))
    error_rate['iter'].append(KOS_estimates(B, Users, Objects, False))
    error_rate['iter_norm'].append(KOS_estimates(B, Users, Objects, True))
    """
    # start iteratively removing workers
    for _ in range(1, limit + 1):
        # hard penalty algorithm
        if mode == 'hard':
            # create conflict graph B^cs
            (conflict_graph, conflict_set) = create_conflict_graph(Objects)
            if conflict_set == {}: 
                break
            # compute optimal semi-matching (OSM) on B^cs
            compute_rep_scores(conflict_graph, Users, conflict_set)
            # determine worker with largest degree in OSM
            high = nlargest(1, range(n), key=lambda i: Users[i].degM)[0]
            assert (Users[high].degM > 0)
        # soft penalty algorithm
        else:
            # compute soft-penalty for each worker
            penalty_spread_iter(Users, Objects, mode)
            # compute worker with largest penalty
            high = nlargest(1, range(n), key=lambda i: Users[i].reputation)[0]
            if Users[high].reputation <= 0:
                break
            assert (Users[high].reputation > 0)

        Users[high].filtered = True
        # add worker to filtered list
        flagged.append(high)

        # remove labels of the filtered worker
        for obj in B.neighbors(high):
            if Objects[obj].conflict:
                if Objects[obj].votes[high] > 0:
                    Objects[obj].upvotes -= 1
                else:
                    Objects[obj].downvotes -= 1
                Objects[obj].votes[high] = 0

        # update conflict set of objects
        for obj in Objects:
            if Objects[obj].upvotes == 0 or Objects[obj].downvotes == 0:
                Objects[obj].conflict = False

        # reset penalties and degrees in OSM
        for u in Users:
            Users[u].degM = 0
            Users[u].reputation = 0.

        # compute error rates of algorithms considering labels of only REMAINING workers
        truth_hat = []
        mv =  majority_voting(Users, Objects)
        predited_truth = get_predicted_truth(Users, Objects)
        truth_hat.append(predited_truth)
        error_rate['pen_maj'].append(mv)
        """
        error_rate['pen_spec_em'].append(spectral_EM_estimates(Users, Objects))
        error_rate['pen_em'].append(EM_estimates(Users, Objects))
        error_rate['pen_iter'].append(KOS_estimates(B, Users, Objects, False))
        error_rate['pen_iter_norm'].append(KOS_estimates(B, Users, Objects, True))
        """
    for alg in ['maj']:#, 'em', 'iter', 'iter_norm', 'spec_em']:
        base_acc = np.array(error_rate[alg])
        base_percent_acc = np.mean(base_acc, axis=1)
        print ('baseline for %s: %s' % (alg, base_percent_acc))
        with_pen_acc = np.array(error_rate['pen_' + alg])
        if len(with_pen_acc) > 0: 
            with_pen_percent_acc = np.mean(with_pen_acc, axis=1)
            number_of_times_better = np.mean(with_pen_percent_acc >= base_percent_acc)
            best_index = np.argmax(with_pen_percent_acc)
            print ('with %s penalty: %s(removing %d workers)' % (mode, with_pen_percent_acc[best_index], best_index + 1))
            print ('relative t-test p values: %s' % ttest_rel(base_acc, with_pen_acc, axis=1)[1][best_index])
            print ('Filtering workers is better %.3f fraction of the time' % number_of_times_better)
            print ('================')
            
            return  with_pen_percent_acc[best_index] , truth_hat[best_index]
    


def real_experiments(dataset, mode):
    (Users, Objects, B) = tweets_sentiments()
    real_world_performance(Users, Objects, B, mode)

def partialy_real_experiments(mode, datafile, truthfile):
    (Users, Objects, B) = tweets_sentiments( datafile, truthfile)
    acc,truth_hat = part_real_world_performance(Users, Objects, B, mode)
    return acc,truth_hat


if __name__ == "__main__":
    datasets = ['dsentiment']
    #algos = ['soft', 'hard']
    algos = ['']
    for (d, a) in itertools.product(datasets, algos):
        print ('Starting dataset: %s and algorithm: %s' % (d, a))
        real_experiments(d, a)
        sys.stdout.flush()
