# -*- coding: utf-8 -*-
"""
Blotto ALLOCATION STOCHASTIC OPTIMISATION MODEL
===============================================
Created on Tue Mar 8 2022
Qiao Yan Soh sohqiaoyan@gmail.com

"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd 

# ===== Model function definitions 

def ObjectiveFx(m):
    """ Maximise score over all scenario """
    return sum(m.GameScore[S] for S in m.S)
                   
def Allocation(m):
    """ Total soldiers is 100 """
    return sum(m.Soldiers[i] for i in m.i) == 100

# ----- ALLOCATION BASED WINNING
def DefineWin1(m, i, S):
    return m.Soldiers[i] - m.Opponents[i, S] <= 200 * m.Overcome[i, S] 

def DefineWin2(m, i, S):
    return m.Soldiers[i] - m.Opponents[i, S] >= 200 * (m.Overcome[i, S] - 1) + m.Overcome[i, S]

# ----- SCORING FUNCTIONS
def ScoreValue(m, i):
    """ Index numbering start at 0, castle values start at 1. """ 
    return m.Value[i] == i + 1

def SingleGameScore1(m, S):
    return m.GameScore[S] == sum(m.score[i, S] for i in m.i)

def SingleGameScore2(m, i, S):
    return m.score[i, S] <= 20 * m.Win[i, S]

def SingleGameScore3(m, i, S):
    return m.score[i, S] >= 20 * (m.Win[i, S] - 1) + m.Value[i]

def SingleGameScore4(m, i, S):
    return m.score[i, S] <= m.Value[i]

# ----- CONSECUTIVE WIN CONSTRAINTS 
def ConsecDummyDef(m, i, S):
    if i < 2:
        return  m.BinarySum[i, S] == sum(m.Overcome[n, S] for n in range(0, i+1))
    else:
        return m.BinarySum[i, S] == m.Overcome[i, S] + m.Overcome[i-2, S] + m.Overcome[i-1, S]

def Consec1(m, i, S):
    return m.BinarySum[i, S]  <= 10 *  m.ConsecFlag[i, S] + 2

def Consec2(m, i, S):
    return m.BinarySum[i, S] >= 10 * (m.ConsecFlag[i, S] - 1) + 3

# ----- WIN DEFINITION
def OverallConsecFlag(m, i, S):
    return m.Flag[i, S] == sum(m.ConsecFlag[n, S] for n in range(0, i+1))

def Win1(m, i, S):
    return m.Win[i, S] >= m.Overcome[i, S]

def Win2(m, i, S):
    return m.Win[i, S] >= m.Flag[i, S] / 10

def Win3(m, i, S):
     return m.Win[i, S] <= m.Overcome[i, S] + m.Flag[i, S] 



# ========== COMPUTATIONAL FUNCTIONS 

def RunModel(n = 1000):
    """ 
    Wrapper for running the entire simulation. 
    """
    m = CreateAbstractModel()           # Create an abstract model
    Opponents = GenerateOpponents(n)    # Generate enemy scenarios
    d = Data(Opponents)                 # Put inputs into appropriate format
    
    return SolveInstance(d, m)          # Solve problem. 


def CreateAbstractModel():
    """ Wrapper function for creating an abstract model, to be populated with input data. """
    m = pyo.AbstractModel()
    
    # ----- Sets
    m.S = pyo.Set(ordered = True, doc = 'Opponent number')
    m.i = pyo.Set(ordered = True, doc = 'Castle number', initialize = list(range(10)))
    
    # ----- Parameters
    m.Opponents = pyo.Param(m.i, m.S, within = pyo.NonNegativeReals)
    
    # ----- Continuous variables (Should actually be integers)
    m.GameScore = pyo.Var(m.S, within = pyo.NonNegativeReals)
    m.Value = pyo.Var(m.i, within = pyo.NonNegativeReals)
    m.score = pyo.Var(m.i, m.S, within = pyo.NonNegativeReals)
    m.BinarySum = pyo.Var(m.i, m.S, bounds=(0,3), within = pyo.NonNegativeReals)
    
    # ----- Binary variables
    m.Win = pyo.Var(m.i, m.S, within = pyo.Binary)
    m.Flag = pyo.Var(m.i, m.S, within =pyo.Binary)
    m.Overcome = pyo.Var(m.i, m.S, within = pyo.Binary)
    m.ConsecFlag = pyo.Var(m.i, m.S, within = pyo.Binary)
    
    # ----- Decision variable
    m.Soldiers = pyo.Var(m.i, within = pyo.NonNegativeIntegers)
    
    # ----- Objective
    m.Obj = pyo.Objective(rule = ObjectiveFx, sense = pyo.maximize)
    
    # ----- Constraints
    m.ValueAllocation = pyo.Constraint(m.i, rule = ScoreValue)
    m.IndivScore1 = pyo.Constraint(m.S, rule = SingleGameScore1)
    m.IndivScore2 = pyo.Constraint(m.i, m.S, rule = SingleGameScore2)
    m.IndivScore3 = pyo.Constraint(m.i, m.S, rule = SingleGameScore3)
    m.IndivScore4 = pyo.Constraint(m.i, m.S, rule = SingleGameScore4)
    
    m.P = pyo.Constraint(m.i, m.S, rule = ConsecDummyDef)
    m.P1 = pyo.Constraint(m.i, m.S, rule = Consec1)
    m.P2 = pyo.Constraint(m.i, m.S, rule = Consec2)
    
    m.W1 = pyo.Constraint(m.i, m.S, rule = OverallConsecFlag)
    m.W2 = pyo.Constraint(m.i, m.S, rule = Win1)
    m.W3 = pyo.Constraint(m.i, m.S, rule = Win2)
    m.W4 = pyo.Constraint(m.i, m.S, rule = Win3)
    
    m.Allocate = pyo.Constraint(rule = Allocation)
    m.WinDef1 = pyo.Constraint(m.i, m.S, rule = DefineWin1)
    m.WinDef2 = pyo.Constraint(m.i, m.S, rule = DefineWin2)
   
    return m


def Data(Opponents,):
    """ 
    Compiles the opponent data into a dictionary format necessary for pyomo to read. 
    INPUTS:
        Opponents : Dictionary with tuple keys of (Castle, Opponent) corresponding to enemy allocations for each castle. 
    """
    n_Opponents = int(len(Opponents) / 10)
    dct = {None: {
        'i': {None: list(range(0, 10))},
        'S': {None: list(range(n_Opponents))},
        'Opponents': Opponents,
        }}
    
    return dct


def SolveInstance(Data, m):
    """
    Builds an instance of the model m with the given data and solves it. 
    Outputs a dictionary with the solution under the 'Sol' key. 
    INPUTS:
        Data : Compiled data 
        m : Abstract model object
    """
    instance = m.create_instance(Data)
    
    opt = pyo.SolverFactory('cplex')            
    opt_results = opt.solve(instance, tee = True)
    
    Opp = pd.DataFrame.from_dict(instance.Opponents.extract_values(), orient = 'index', columns = ['Enemies'])
    Opp.index = pd.MultiIndex.from_tuples(Opp.index, names = ['Castle', 'Opponent'])
    Opp.reset_index(inplace = True)
    Opp = Opp.pivot(index = 'Castle', columns = 'Opponent')
    Sol = pd.DataFrame.from_dict(instance.Soldiers.extract_values(), orient = 'index', columns = ['Soldiers'])
    
    return {'Instance': instance, 'Opp': Opp, 'Sol':Sol, 'Solver' : opt_results, }


def GenerateOpponents(n = 1000):
    """ 
    Randomly generates n scenarios of possible soldier allocations. 
    Outputs a dictionary containing soldier allocations for each (Castle, Opponent). 
    INPUTS: 
        n : number of opponents to generate. 
    """
    O_Set = []                      # Storage list for opponent arrays
    while len(O_Set) < n:           # Continue generating opponents until the desired number is reached
        a = np.random.random(10)    # Generate 10 random numbers between [0,1]
        a = a/sum(a) * 100          # Scale to 100
        b = np.round(a)             # Round floats to integer values. 
        
        difference = sum(b) - 100           # Calculate any over/undershoots of 100 soldiers. 
        b = SelectRandom(b, difference)     # Adjust a random allocation to fit 100 soldier constraint. 
        O_Set.append(b)                     # Add opponent to total list. 
        
    dct = dict()            # Adjust list of opponent allocations into pyomo appropriate format. 
    count = 0
    for array in O_Set: 
        dct[count] = array
        count += 1
        
    df = pd.DataFrame(dct)
    df = df.unstack()
    df = df.swaplevel()
    
    return dict(df)

def SelectRandom(b, difference):
    """ 
    Selects a random castle allocation value to adjust and checks that the value being adjusted is feasible. 
    Outputs an adjusted array that sums to 100. 
    INPUTS:
        b : array to be adjusted 
        difference : value difference between original input array and 100 
    """
    idx = np.random.randint(10)     # Select a random index
    if b[idx] >= difference :       # if the selected allocation value is larger than the difference
        b[idx] -= difference        # adjust the selected allocation value. 
    else:                           # Otherwise, select a different allocation value. 
        SelectRandom(b, difference)
    return b


