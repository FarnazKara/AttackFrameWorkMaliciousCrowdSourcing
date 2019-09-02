import random
import os
import csv
import operator
import networkx as nx
from networkx.algorithms import bipartite
from operator import itemgetter

"""
1 Find an initial semi-matching, M.
2 While there exists a cost-reducing path, P
3 Use P to reduce the cost of M.

"""

class MVHard:

    def __init__(self,e2wl,w2el,label_set, truthpath):

        self.e2wl = e2wl
        self.w2el = w2el
        self.workers =  self.w2el.keys()
        self.label_set = label_set
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
        
        
    
    def createBipartite(self,task_conflict_set):
        nodes = [t+"_pos" for t in task_conflict_set] + [t+"_zero" for t in task_conflict_set]
        B = nx.Graph()


        B.add_nodes_from(nodes, bipartite= 0 )
        B.add_nodes_from(self.workers, bipartite=1) # Add the node attribute "bipartite"
        
        for t in task_conflict_set:
            for value in self.e2wl[t]:
                    w = value[0]
                    findans = [list_t[1] for list_t in self.w2el[w] if list_t[0] == t]
                    if findans == ['1.0'] or findans == ['1']:
                        B.add_edges_from([(w, t+"_pos")])
                    elif findans == ['0.0'] or findans == ['0']: 
                        B.add_edges_from([(w, t+"_zero")])
        return B
            
    def getdifferenceEdgesGraph(self,refrencedGraph, other ):
        diff = []
        g = nx.Graph()
        g.add_nodes_from(refrencedGraph.nodes())
        for edge in refrencedGraph.edges():
            if not (other.has_edge(edge[0],edge[1])) or not (other.has_edge(edge[1],edge[0])):
                diff.append([edge[0],edge[1]])
        g.add_edges_from(diff)
        return g
        
    
    def buildforest_findCRP(self, init_osm,bgraph):
        root = []  
        seen = []
        ncpr = 0
        cprs = {}

        diffEdgesgraph = self.getdifferenceEdgesGraph(bgraph, init_osm)
        # U nodes has tasks and V nodes has users        
        if not nx.is_connected(bgraph):
            print( 'Disconnected graph: Ambiguous solution for bipartite sets.')
            UNodes = set(n for n,d in bgraph.nodes(data=True) if d['bipartite']==0)
            VNodes = set(bgraph) - UNodes
        else:
            UNodes, VNodes  = bipartite.sets(bgraph)

        vdeg_dict = dict.fromkeys(VNodes)
        
        for v in VNodes: 
            cprfound = False
            if len(init_osm[v]) == 0:
                del vdeg_dict[v]
            else: 
                vdeg_dict[v] = len(init_osm[v])
                
        if max(vdeg_dict.values()) - min(vdeg_dict.values()) <= 1: 
            return cprs
        else:
            #least loaded nodes in Vnodes
            VNodes = sorted(vdeg_dict, key = vdeg_dict.get, reverse = True)
            for v in VNodes:
                print("Im here not stack")

                if len(init_osm[v]) == 1: #it does not have neighbor
                    continue
                
                # v to u should pick from M graph but u to v should pick from the edges not in M
                #remainerVNodes = list(set(VNodes) - set(seen))
                remainVDeg = vdeg_dict
                for rn in seen:
                    if rn in remainVDeg:
                        del remainVDeg[rn]
                
                if max(remainVDeg.values()) - min(remainVDeg.values()) <= 1: 
                    print("ha ha ")
                    return  cprs
                
                nodeseen = []
                # The other v nodes in previous cprs 
                if v not in seen:
                    
                   #path = []
                   stack = []
                   vcpr = []
                   ucpr = []
                   cur_root = v
                   root.append(v)
                   
                   stack.append([-1, v])
                     
                   while stack:
                       parent, cur = stack.pop()
                       while stack: 
                            if cur and cur in seen:
                                parent, cur = stack.pop()
                            elif cur and cur in nodeseen:
                                parent, cur = stack.pop()
                            else:
                                break
                       nodeseen.append(cur)
                       if cur in VNodes:
                           if cur in vcpr:
                               print("I have already see this guy in v")
                           else: 
                               neighbors = init_osm[cur]  
                               pcur = parent
                               while len(neighbors) == 0 and  pcur == parent and cur in VNodes and len(stack) > 0:
                                   if cur != cur_root:
                                       if  len(init_osm[cur]) < (len(init_osm[cur_root]) - 1):
                                           print("CRP Path is found\n")
                                           if cur not in vcpr:
                                               vcpr.append(cur)
                                           cprs[ncpr] = [vcpr, ucpr] 
                                           seen = vcpr
                                           vcpr = []
                                           ucpr = []
                                           stack = []
                                           ncpr += 1
                                           cprfound = True
                                           return cprs
                                           break
                                           
                                       else:
                                           parent, cur = stack.pop()
                                           neighbors = init_osm[cur]
                                   else:
                                        print("it is alone no cpr")
                                        break
                                   
                               if cprfound == False and pcur != parent: 
                                   print("It could not find the cpr in this tree")
                                   idx = ucpr.index(parent)
                                   ucpr = ucpr[0:idx]
                                   vcpr = vcpr[0:idx]
                                   stack.append([parent, cur])                                                                      
                                   nodeseen = vcpr + ucpr
                                   
                               elif cur in UNodes:
                                   print("It could not find the cpr in this tree")
                                   idx = vcpr.index(parent)
                                   ucpr = ucpr[0:idx]
                                   vcpr = vcpr[0:idx+1]
                                   stack.append([parent, cur])                                                                      
                                   nodeseen = vcpr + ucpr

                               else:
                                   #if parent == pcur:
                                       #if len(vcpr)> 0:
                                        #   vcpr.pop()
                                   vcpr.append(cur)
                                   #print("add neighbor for :")
                                   #print(cur)
                                   for n in neighbors: 
                                       if n not in nodeseen:
                                           stack.append([cur, n])
                                   nodeseen.append(cur)
                                   
                       elif cur in UNodes: 
                            if cur in ucpr:
                               print("I have already see this guy in U")
                            else: 
                               neighbors = diffEdgesgraph[cur]
                               pcur = parent 
                               while len(neighbors) == 0  and pcur == parent and cur in UNodes and len(stack) > 0:
                                   parent, cur = stack.pop()
                                   neighbors = diffEdgesgraph[cur]
                               if pcur != parent:
                                   print("It could not find the cpr in this tree")
                                   idx = vcpr.index(parent)
                                   print(idx)
                                   ucpr = ucpr[0:idx]
                                   vcpr = vcpr[0:idx+1]
                                   stack.append([parent, cur])
                                   nodeseen = vcpr + ucpr

                                   
                               if cur in VNodes:
                                   print("It could not find the cpr in this tree")
                                   
                                   print(cur)
                                   print('parent')
                                   print(parent)
                                   print(len(ucpr))
                                   print(len(stack))
                                   idx = ucpr.index(parent)
                                   print(idx)
                                   ucpr = ucpr[0:idx+1]
                                   vcpr = vcpr[0:idx+1]
                                   stack.append([parent, cur])
                                   nodeseen = vcpr + ucpr

                                   
                               else: 
                                   if parent == pcur:
                                       if len(ucpr) > 0: 
                                           ucpr.pop()
                                   ucpr.append(cur)
                                   for n in neighbors: 
                                       if n not in  nodeseen:
                                           stack.append([cur, n])
                                   nodeseen.append(cur) 
                       
                       if cprfound == False and cur in VNodes and len(init_osm[cur]) < (len(init_osm[cur_root]) - 1):
                           print("CRP Path is found\n")
                           if cur not in vcpr:
                               vcpr.append(cur)
                           cprs[ncpr] = [vcpr, ucpr] 
                           seen = vcpr
                           vcpr = []
                           ucpr = []
                           stack = []
                           nodeseen = []
                           ncpr += 1
                           return cprs
                           break
                #seen = list(set(seen + vcpr))
                cprfound = False
            return cprs
                
                
        
    def findInitialOptimalSMatching(self, bgrapg):
        if nx.is_connected(bgrapg) == False:
            print("the first graph is not connected")
        else:
            print("the first graph is  connected!!:)")
        
        
        
        # top: U --> tasks  bottom : V --> users
        if not nx.is_connected(bgrapg):
            print( 'Disconnected graph: Ambiguous solution for bipartite sets.')
            top_nodes = set(n for n,d in bgrapg.nodes(data=True) if d['bipartite']==0)
            bottom_nodes = set(bgrapg) - top_nodes
        else:
            top_nodes, bottom_nodes = bipartite.sets(bgrapg)
        data = dict(bgrapg.degree(top_nodes))
        Unode_sorted = sorted(data, key=data.get,reverse=True) #the U-vertices are sorted by increasing degree.
        
        semiB = nx.Graph()
        semiB.add_nodes_from(top_nodes, bipartite=0) # Add the node attribute "bipartite"
        semiB.add_nodes_from(bottom_nodes, bipartite=1)
        #Match each u \in U with its least-loaded V-neighbor.
        for u in Unode_sorted:
            neighbours = list(bgrapg[u])
            semi_vdeg = dict(semiB.degree(neighbours))
            semi_vnode_sorted = sorted(semi_vdeg, key=semi_vdeg.get) 
            freq = [ node_name for node_name,value_deg in semi_vdeg.items() if value_deg == semi_vdeg[semi_vnode_sorted[0]] ] 
            
            if len(semiB.edges) == 0:
                vdeg = dict(bgrapg.degree(freq))
                vnode_sorted = sorted(vdeg, key=vdeg.get) 
                #Each U-vertex is then considered in turn, and assigned to a V-neighbor with least load. 
                semiB.add_edges_from([(u,vnode_sorted[0])])
            elif len(freq) > 1:             
                vdeg = dict(bgrapg.degree(freq))
                vnode_sorted = sorted(vdeg, key=vdeg.get) 
                #Each U-vertex is then considered in turn, and assigned to a V-neighbor with least load. 
                semiB.add_edges_from([(u,vnode_sorted[0])])
            else: 
                semiB.add_edges_from([(u,semi_vnode_sorted[0])])
        if nx.is_connected(semiB) == False: 
            print("The semi graph is not connected")
        return semiB 
    
    
    def findCostReducingPaths(self,forest, unode, semimatch):
        cost = []
        for root, path in forest.items():
            subcost = 0
            for node in path: 
                if node in unode: subcost += len(semimatch[node])
            cost.append((root,subcost))
        
        idx_root = cost.index(min(cost))
        root = list(forest.keys())[idx_root]
        return {root:forest[root]}
    
    def get_representative_w(self, g_osm, task):
        tpos = task+"_pos"
        tzero = task+"_zero"
        wlist = g_osm[tpos]       
        deg_w = [(w,len(g_osm[w])) for w in wlist]
        idx_pos = deg_w.index(max(deg_w , key=itemgetter(1)))
        wpos = deg_w[idx_pos][0]
        
        wlist = g_osm[tzero]
        deg_w = [(w,len(g_osm[w])) for w in wlist]
        idx_zero = deg_w.index(max(deg_w , key=itemgetter(1)))
        
        return  wpos, deg_w[idx_zero][0]
    
    def removeCostReducingPaths(self,crps_list, init_gosm):
        for key, crp in crps_list.items():
            ucrp = crp[1]
            vcrp = crp[0]
            redges = []
            aedges = []
            cnt = len(ucrp)
            if len(ucrp) >= len(vcrp):
                cnt = len(vcrp) - 1
            for i in range(cnt): 
                redges.append((vcrp[i], ucrp[i]))
                aedges.append((ucrp[i], vcrp[i+1]))
            init_gosm.remove_edges_from(redges)
            init_gosm.add_edges_from(aedges)
        return init_gosm
    
    def Run(self):
        diff_acc = []
        e2lpd = self.runMV()
        #truthfile = os.getcwd() + "\\data\\truth.csv"
        diff_acc.append(self.getaccuracy(self.truthfile, e2lpd,self.label_set))
        for i in range(1,10):
            e2wl = self.e2wl
            p2w = {}
            task_conflict_set = self.getConflictedSet()
            #just for test
            #task_conflict_set = ['task_0','task_1','task_2','task_3','task_4']
            if len(task_conflict_set) == 0:
                print ("There is no any conflicted task")
                return diff_acc[0], diff_acc[0], 0
                
            bipartite_g = self.createBipartite(task_conflict_set)
            init_osm = self.findInitialOptimalSMatching(bipartite_g)
            
            #findCostReducingPaths
            oldcrps = {}
            allcrps = []
            while True:
                crps = self.buildforest_findCRP(init_osm,bipartite_g)
                #crps = self.findCostReducingPaths(forest)
                if len(crps) == 0 or oldcrps == crps or crps in allcrps:
                    break
                osm  =  self.removeCostReducingPaths(crps, init_osm)
                init_osm = osm
                oldcrps = crps
                allcrps.append(list(crps.values()))
                
            
            for t in task_conflict_set:
                w_pos , w_zero = self.get_representative_w(init_osm, t)
                if w_pos in p2w.keys():
                    p2w[w_pos] += len(init_osm[w_pos])
                else:
                    p2w[w_pos] = len(init_osm[w_pos])
                    
                if w_zero in p2w.keys():
                    p2w[w_zero] += len(init_osm[w_zero])
                else: 
                    p2w[w_zero] = len(init_osm[w_zero])
            
            selectedworker = self.worker_highestPenalty(p2w)
            self.updateworkerTaskList(selectedworker)
            e2lpd = self.runMV()
            truthfile = self.truthfile # os.getcwd() + "\\data\\truth.csv"
            diff_acc.append(self.getaccuracy(truthfile, e2lpd,self.label_set))

            
            """
            for t in task_conflict_set:
                labelspecifictask = []
                labelspecifictask = [e2wl[t][i][1] for i in range(len(e2wl[t]))]
                cntLabels = Counter(labelspecifictask)
                for value in e2wl[t]:
                    w = value[0]
                    findans = [list_t[1] for list_t in w2el[w] if list_t[0] == t]
                    if findans == ['1']:
                        Stw = 1.0/cntLabels['1']
                        Stw = float("{0:.3f}".format(Stw))

                        if w in p2w:
                            p2w[w] = p2w[w] + [Stw]
                        else:
                            p2w[w] =  [Stw]
                    elif findans == ['0']:
                        Stw = 1.0/cntLabels['0']
                        Stw = float("{0:.3f}".format(Stw))
                        if w in p2w:
                            p2w[w] = p2w[w] + [Stw]
                        else:
                            p2w[w] = [Stw]
            
            for key , value in p2w.items():
                 workerPenalty[key] = float("{0:.3f}".format(mean(value))) 
        
            selectedworker = self.worker_highestPenalty(workerPenalty)
            self.updateworkerTaskList(selectedworker)
            e2lpd = self.runMV()
            truthfile = os.getcwd() + "\\truth.csv"
            diff_acc.append(self.getaccuracy(truthfile, e2lpd,self.label_set))
        """
        return diff_acc[0], max(diff_acc), diff_acc.index(max(diff_acc))      
        
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
    
            if truth.split('.')[0]== e2truth[e].split('.')[0]:
                tcount += 1
    
        return tcount*1.0/count
    
    
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
    datafile = os.getcwd() + "\\answers.csv"
    truthfile = os.getcwd() + "\\data\\truth.csv"

    e2wl,w2el,label_set = gete2wlandw2el(datafile)
    firstacc, bestacc, index = MVHard(e2wl,w2el,label_set, truthfile).Run()

    print(firstacc)
    print(bestacc)
    print(index)
    

