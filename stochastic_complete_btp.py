import gurobipy as gb
import numpy as np
from gurobipy import GRB
from collections import defaultdict
import pandas as pd

df = pd.read_csv('data.csv')
df.drop(9, inplace=True)
returns = pd.read_csv('samples.csv')
returns.drop(['Unnamed: 0'], inplace=True, axis=1)

options = pd.read_csv('options.csv')
 
c0 = ['inr', 'jpy', 'usd']
c = ['jpy', 'usd']

alpha = 0.95

delta_stock = 0.00125
delta_forex = 0.00125
#https://www.mathworks.com/help/finance/portfolio.setcosts.html

S_0 = defaultdict(list)
S_0['inr'] = df['S_0'][:5]
S_0['jpy'] = df['S_0'][5:8]
S_0['usd'] = df['S_0'][8:]

r = defaultdict(list)
r['inr'] = returns.iloc[:,:5]
r['jpy'] = returns.iloc[:,5:8]
r['usd'] = returns.iloc[:,8:10]

o_i = returns.iloc[:,10:13]
o_u = returns.iloc[:,13:16]

b = defaultdict(list)
ec = defaultdict()
h0 = defaultdict()
fc = defaultdict()

ec['usd'] = 81
ec['jpy'] = 0.64
ec['inr'] = 1

fc['jpy'] = 0.61
fc['usd'] = 82.02

for c in c0:
    b[c] = [0]*len(S_0[c])
    h0[c] = 1000/ec[c]
    
    
val = []

for T in range(1, 100):

    S_T = defaultdict(list)
    
    O_S_I = []
    O_S_U = []
    
    for i in range(1000):
        O_S_I.append(list(np.repeat([round(a*b, 2) for a,b in zip(options['S_I'].tolist(), pow(1 + o_i.loc[i], T).tolist())], 2)))
        O_S_U.append(list(np.repeat([round(a*b, 2) for a,b in zip(options['S_U'].tolist(), pow(1 + o_u.loc[i], T).tolist())], 2)))
        for key in S_0.keys():
            S_T[key].append([round(a*b, 2) for a,b in zip(S_0[key].tolist(), pow(1 + r[key].loc[i], T).tolist())])
    
    #defining the model
    
    N = 1000
    p = 1/N
    #probability of occurence of each scenario
    model = gb.Model('mipl')
        
    #https://support.gurobi.com/hc/en-us/community/posts/360073717212-Array-or-List
        
    assets = [('inr', 0), ('inr', 1), ('inr', 2), ('inr', 3), ('inr', 4), ('jpy', 0), ('jpy', 1), ('jpy', 2), ('usd', 0), ('usd', 1)]
        
    ce = ['jpy', 'usd']
    
    V_0 = 0
    
    for key in h0.keys():
        V_0 += ec[key]*(h0[key] + np.dot(b[key], S_0[key]))
    #value of the initial portfolio
    
    z = model.addVar(vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "z")
    ufc = {(i): model.addVars(ce, vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "ufc") for i in range(1000)}
    
    R = model.addMVar(shape = N, vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "R")
    L = model.addMVar(shape = N, vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "L")
    y = model.addMVar(shape = N, vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "y")
    V_T = model.addMVar(shape = N, vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "V_T")
    
    xic = {(i): model.addVars(assets, vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "xic") for i in range(1000)}
    #units of asset i in Ic of currency c in C0 purchased
    vic = {(i): model.addVars(assets, vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "vic") for i in range(1000)}
    #units of asset i in Ic of currency c in C0 sold
    wic = {(i): model.addVars(assets, vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "wic") for i in range(1000)}
    #units of asset i in Ic of currency c in C0 held in the revised portfolio
    
    xec = {(i): model.addVars(ce, vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "xec") for i in range(1000)}
    #amount of the base currency exchanged in the spot market for foreign currency c in C
    vec = {(i): model.addVars(ce, vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "vec") for i in range(1000)}
    #amount of the base currency collected from the spot sale of foreign currency c in C
    
    numc = {(i): model.addVars(['inr', 'usd'], [0, 1, 2, 3, 4, 5], vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "nc") for i in range(1000)}
    nump = {(i): model.addVars(['inr', 'usd'], [0, 1, 2, 3, 4, 5], vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "np") for i in range(1000)}
    
    for i in range(1000):    
        model.addConstr(h0['inr'] + (1 - delta_stock)*np.sum([vic[i]['inr', k] for k in range(0, 5)]*S_0['inr']) + (vec[i].sum())*(1 - delta_forex) == (1 + delta_stock) * np.sum([xic[i]['inr', k] for k in range(0, 5)]*S_0['inr']) + (xec[i].sum())*(1 + delta_forex) + np.sum(options['KC_I']*[numc[i]['inr', k] for k in range(0, 6)]) + np.sum(options['KP_I']*[nump[i]['inr', k] for k in range(0, 6)]))
        model.addConstr(h0['usd'] + np.sum([vic[i]['usd', k] for k in range(0, 2)]*S_0['usd'])*(1 - delta_stock) + xec[i]['usd']/ec['usd'] == np.sum([xic[i]['usd', k] for k in range(0, 2)]*S_0['usd'])*(1 + delta_stock) + vec[i]['usd']/ec['usd'] + np.sum(options['KC_U']*[numc[i]['usd', k] for k in range(0, 6)]) + np.sum(options['KP_U']*[nump[i]['usd', k] for k in range(0, 6)]))
        model.addConstr(h0['jpy'] + np.sum([vic[i]['jpy', k] for k in range(0, 3)]*S_0['jpy'])*(1 - delta_stock) + xec[i]['jpy']/ec['jpy'] == np.sum([xic[i]['jpy', k] for k in range(0, 3)]*S_0['jpy'])*(1 + delta_stock) + vec[i]['jpy']/ec['jpy'])
        #cash balance constraints for the initial portfolio
        
        #finding the value of the final portfolio
        model.addConstr(V_T[i] == sum([a*b for a, b in zip([wic[i]['inr', k] for k in range(0, 5)],S_T['inr'][i])]) + ec['jpy']*(sum([a*b for a, b in zip([wic[i]['jpy', k] for k in range(0, 3)],S_T['jpy'][i])])) + ec['usd']*(sum([a*b for a, b in zip([wic[i]['usd', k] for k in range(0, 2)],S_T['usd'][i])])) + ufc[i]['jpy'] + ec['jpy']*(- ufc[i]['jpy']/fc['jpy']) + ufc[i]['usd'] + ec['usd']*( - ufc[i]['usd']/fc['usd'] + np.sum([numc[i]['usd', k] for k in range(0, 6)]*np.maximum(np.array(O_S_U[i] - options['C_U']), np.zeros(6))) + np.sum([nump[i]['usd', k] for k in range(0, 6)]*np.maximum(np.array(options['P_U'] - O_S_U[i]), np.zeros(6)))) + np.sum([numc[i]['inr', k] for k in range(0, 6)]*np.maximum(np.array(O_S_I[i] - options['C_I']), np.zeros(6))) + np.sum([nump[i]['inr', k] for k in range(0, 6)]*np.maximum(np.array(options['P_I'] - O_S_I[i]), np.zeros(6))))
        
        model.addConstr(ufc[i]['jpy'] <= p*ec['jpy'] * sum([a*b for a, b in zip([wic[i]['jpy', k] for k in range(0, 3)], S_T['jpy'][i])]))
        model.addConstr(ufc[i]['usd'] <= p*ec['usd'] * sum([a*b for a, b in zip([wic[i]['usd', k] for k in range(0, 2)], S_T['usd'][i])]))
        
        model.addConstr(np.sum([nump[i]['inr', k] for k in range(0, 6)]*np.array(options['KP_I'])) + np.sum([numc[i]['inr', k] for k in range(0, 6)]*np.array(options['KC_I'])) <= np.sum([wic[i]['inr', k] for k in range(0, 5)]*S_0['inr']))
        model.addConstr(np.sum([nump[i]['usd', k] for k in range(0, 6)]*np.array(options['KP_U'])) + np.sum([numc[i]['usd', k] for k in range(0, 6)]*np.array(options['KC_U'])) <= np.sum([wic[i]['usd', k] for k in range(0, 2)]*S_0['usd']))
        
        model.addConstr(R[i] == V_T[i]/V_0 - 1)
        model.addConstr(L[i] == -R[i])
        model.addConstr(y[i] + z >= L[i])
        
        for key in S_0.keys():
           for k in range(0, 6):
                if key != 'jpy': #removing options
                   model.addConstr(numc[i][key, k] == 0)
                   model.addConstr(nump[i][key, k] == 0)
                
           if key != 'inr': #removing currency hedging
               model.addConstr(ufc[i][key] == 0)
               model.addConstr(xec[i][key] == 0)
               model.addConstr(vec[i][key] == 0)    
        
        for key in S_0.keys():
            indi = [0]*len(S_0[key])
            for k in range(0, len(indi)):
                indi[k] = 1
                model.addConstr(np.sum([wic[i][key, q] for q in range(0, len(S_0[key]))]*indi[k]) == b[key][k] + np.sum([xic[i][key, q] for q in range(0, len(S_0[key]))]*indi[k]) - np.sum([vic[i][key, q] for q in range(0, len(S_0[key]))]*indi[k]))
                model.addConstr(np.sum([vic[i][key, q] for q in range(0, len(S_0[key]))]*indi[k]) <= b[key][k])
                indi[k] = 0
    
        
    model.addConstr(sum(R)*p >= 1)
    #minimum expected return
        
    model.setObjective(z + p/(1 - alpha)*sum(y), GRB.MINIMIZE)
    #linear objective function
        
    model.optimize()
    val.append(model.objVal)

df = pd.DataFrame(val)
df.to_csv('only_stocks.csv')