import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

import os
import operator
import random, math, gym
import pandas as pd
import numpy as np


from utils_results import ResultFile
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils import ExperienceReplay

class Brain:
    def __init__(self, num_state, num_action, configDict, RL_GAMMA = 0.99):
        
        self.set_model(configDict)

        self.num_state = num_state
        self.num_action = num_action
        
        self.lr = configDict["learning_rate"] if "learning_rate" in configDict else 0.00025
        self.model = self._createModel()
        
        # self.cp_callback = keras.callbacks.ModelCheckpoint(self.cp_path, save_weights_only=True,verbose=1)

     
        # parameters for RL algorithm:
        self.GAMMA = RL_GAMMA

    def set_model(self, configDict):
        self.modelFName = configDict["directory"] + "/DRL_UAV_"
        self.modelFName += str(configDict["zDistSuccess"]) + "_" + str(configDict["restartType"])  + "_" + str(configDict["initZOptions"]) + ".h5"
        self.modelFName = self.modelFName.replace(" ","")
    def load(self):
        
        if os.path.isfile(self.modelFName):
            print "loaded ->", self.modelFName
            self.model.load_weights(self.modelFName)
            return True
        else:
            print self.modelFName, "not exist"
            return False

    def save_latest(self):
        self.model.save(self.modelFName)

    def _createModel(self): # model: state -> v(state value)
        model = Sequential()

        model.add(Dense(64, activation='relu', input_dim=self.num_state))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_action, activation='linear'))

        opt = RMSprop(lr=self.lr)
        model.compile(loss='mse', optimizer=opt)
        
        return model
    
    def train(self, x, y, batch_size=64, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def predict(self, batch_states, verbose=0):   # batch prediction!
        # input type: state=[state1, state2,...]  -> type: list!
        # e.g.:
        # [array([-0.02851337,  0.04295018, -0.0197721 , -0.00788878]), array([-0.02851337,  0.04295018, -0.0197721 , -0.00788878])]
        # print("Batch len_state: ",len(batch_states))
        # print("Batch state",batch_states)
        return self.model.predict(batch_states, verbose=verbose)
    
    def predictOne(self, state_test):           # solo prediction! 
        # (You have to reshape the input!!!)
        # input type: state_test                -> type: array!
        # e.g.:
        # [-0.02851337  0.04295018 -0.0197721  -0.00788878]
        # reshape: state_test.reshape(1, self.num_state) =>
        # array([[-0.02851337,  0.04295018, -0.0197721 , -0.00788878]])
        # print("One len_state: ",len(state_test))
        # print("One state",state_test)
        return self.predict(state_test.reshape(1, self.num_state)).flatten()

class AlgoBase(object):
    def __init__(self, num_state, num_action, configDict, createResults=True, numToWrite=10):
        # parameters of External Environment:
        self.num_state = num_state
        self.num_action = num_action
        self.numToWrite = numToWrite
        
        dir2Save = "./" + configDict["directory"] + "/"
        if createResults:
            self.resultFile = ResultFile(dir2Save + "talkerResults", numToWrite=numToWrite)
        else:
            self.resultFile = None
        
        self.episodes = 0
    
    def act(self, state):
        return -1

    def load(self):
        if self.resultFile != None:
            loaded = self.resultFile.Load()
            if loaded:
                self.episodes = self.resultFile.NumRuns()
            print "model num episdoes =", self.episodes

    def observe(self, s,a,r,s_,done):
        pass
    
    def replay(self):
        pass

    def end_episode(self, r, sumR, steps, realR=0.0):
        saved = False
        if self.resultFile != None:
            saved = self.resultFile.end_run(r, sumR, steps, realR)
        
        return saved, ""

    def real_action(self):
        return False
    
    def get_results_name(self, directory, zDistSuccess, restartype, initZ):
        name = "./" + directory + "/" + "talkerResults_" + str(zDistSuccess) + "_" + str(restartype) + "_" + str(initZ)
        return name.replace(" ","")
    
    def next_model(self, configDict, load=False):
        resultFName = self.get_results_name(configDict["directory"], configDict["zDistSuccess"], configDict["restartType"], configDict["initZOptions"])
        
        self.resultFile = ResultFile(resultFName, numToWrite=self.numToWrite)
        if load:
            self.resultFile.Load()
            self.episodes = self.resultFile.NumRuns()
        else:
            self.episodes = 0
    
    def Results(self, size, key="reward"):
        return self.resultFile.Results(size, key)
    
    def NumEpisodesAll(self, configDict):
        allRuns = configDict["runs"]
        numEpisodes = 0
        for run in allRuns:
            resultName = self.get_results_name(configDict["directory"], run[0], run[1], run[2])
            result = ResultFile(resultName)
            result.Load()
            numEpisodes += result.NumRuns()
        
        return numEpisodes



class AlgoDQN(AlgoBase):
    def __init__(self, num_state, num_action, configDict, train=True):
        super(AlgoDQN, self).__init__(num_state, num_action, configDict, createResults=False)
        # parameters of Internal DRL algorithm:
        ## Memory:
        self.MEMORY_CAPACITY = 100000
        self.GAMMA = 0.95
        ## Deep network: 
        self.MEMORY_BATCH_SIZE = 64 # number of data for one training! ?(Maybe we can set MEMORY_BATCH_SIZE = MEMORY_CAPACITY)

        self.train = train
        if train:
            ## RL algorithm:
            ## Random selection proportion:
            self.MAX_EPSILON = 1.0
            self.MIN_EPSILON = 0.01
            self.LAMBDA = 0.005  # speed of decay
            self.epsilon = self.MAX_EPSILON
        else:
            self.epsilon = 0.0

        self.brain = Brain(num_state, num_action, configDict, RL_GAMMA=self.GAMMA)

        self.memory = ExperienceReplay(self.MEMORY_CAPACITY)
        self.next_model(configDict)
    
    
    def next_model(self, configDict, load=False):
        super(AlgoDQN, self).next_model(configDict, load)
        self.brain.set_model(configDict)
    

    def load(self):
        loaded = self.brain.load()
        self.resultFile.Load()
        if loaded:
            self.episodes = self.resultFile.NumRuns()
        print "model num episdoes =", self.episodes

    def act(self, state):   # action:[0,1,2,...,num_action-1]
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_action-1)
        else:
            action = np.argmax(self.brain.predictOne(state_test=state))  # get the index of the largest number, that is the action we should take. -libn
        
        return action

    def observe(self, s,a,r,s_,done):
        self.memory.add(experience)

        # decrease Epsilon to reduce random action and trust more in greedy algorithm

    def end_episode(self, r, sumR, steps, realR):
        self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.episodes)
        self.episodes += 1
        saveModel = self.resultFile.end_run(r, sumR, steps, realR)
        if saveModel:
            self.brain.save_latest()

        return saveModel, ""

    def replay(self):   # get knowledge from experience!
        batch = self.memory.sample(self.MEMORY_BATCH_SIZE)
        # batch = self.memory.sample(self.memory.num_experience())  # the training data size is too big!
        len_batch = len(batch)

        no_state = np.zeros(self.num_state)

        batch_states = np.array([o[0] for o in batch])
        batch_states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch])

        # print('Batch states:')
        # print(batch_states)
        # print('Batch states_:')
        # print(batch_states_)
        
        v = self.brain.predict(batch_states)
        v_ = self.brain.predict(batch_states_)

        # inputs and outputs of the Deep Network:
        x = np.zeros((len_batch, self.num_state))
        y = np.zeros((len_batch, self.num_action))

        for i in range(len_batch):
            o = batch[i]
            s = o[0]; a = int(o[1]); r = o[2]; s_ = o[3]

            v_t = v[i]
            if s_ is None:
                v_t[a] = r
            else:
                v_t[a] = r + self.GAMMA * np.amax(v_[i]) # We will get max reward if we select the best option.

            x[i] = s
            y[i] = v_t

        self.brain.train(x, y, batch_size=len_batch)

class AlgoA2C(AlgoBase):
    def __init__(self, num_state, num_action, configDict, train=True):
        super(AlgoA2C, self).__init__(num_state, num_action, configDict, createResults=False)

        # parameters of Internal DRL algorithm:
        ## Memory:
        self.MEMORY_CAPACITY = 100000
        self.GAMMA = 0.95
        ## Deep network: 
        self.MEMORY_BATCH_SIZE = 64 # number of data for one training! ?(Maybe we can set MEMORY_BATCH_SIZE = MEMORY_CAPACITY)

        self.train = train
        if train:
            ## RL algorithm:
            ## Random selection proportion:
            self.MAX_EPSILON = 1.0
            self.MIN_EPSILON = 0.01
            self.LAMBDA = 0.005  # speed of decay
            self.epsilon = self.MAX_EPSILON
        else:
            self.epsilon = 0.0

        self.brain = Brain(num_state, num_action, configDict, RL_GAMMA=self.GAMMA)

        self.memory = ExperienceReplay(self.MEMORY_CAPACITY)
        self.next_model(configDict)
    
    
    def next_model(self, configDict, load=False):
        super(AlgoA2C, self).next_model(configDict, load)
        self.brain.set_model(configDict)

    def load(self):
        loaded = self.brain.load()
        self.resultFile.Load()
        if loaded:
            self.episodes = self.resultFile.NumRuns()

    def act(self, state):   # action:[0,1,2,...,num_action-1]
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_action-1)
        else:
            action = np.argmax(self.brain.predictOne(state_test=state))  # get the index of the largest number, that is the action we should take. -libn
        
        return action

    def observe(self, s,a,r,s_,done):
        self.memory.add(experience)

        # decrease Epsilon to reduce random action and trust more in greedy algorithm

    def end_episode(self, r, sumR, steps, realR):
        self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.episodes)
        self.episodes += 1
        saveModel = self.resultFile.end_run(r, sumR, steps, realR)
        if saveModel:
            self.brain.save_latest()

        return saveModel, ""

    def replay(self): 
        pass

    def learn(self): 


        size = self.memory.num_experience()
    
        allHist = self.memory.sample(self.memory.num_experience()) 
        no_state = np.zeros(self.num_state)

        s = np.array([o[0] for o in batch])
        s_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch])
        
        a = [int(o[1]) for o in batch]
        r = [int(o[2]) for o in batch]

        notDone = [False if o[3] is None else True for o in batch]
        
        idxHist = np.arange(self.MEMORY_BATCH_SIZE)
        
        v = self.brain.predict(s)
        v_ = self.brain.predict(s_)

        # inputs and outputs of the Deep Network:
        x = np.zeros((size, self.num_state))
        y = np.zeros((size, self.num_action))

        y = r + self.GAMMA * notDone * np.amax(v_)


        for e in numEpochs:


            for i in range(len_batch):
                o = batch[i]
                s = o[0]; a = int(o[1]); r = o[2]; s_ = o[3]

                v_t = v[i]
                if s_ is None:
                    v_t[a] = r
                else:
                    v_t[a] = r + self.GAMMA * np.amax(v_[i]) # We will get max reward if we select the best option.

                x[i] = s
                y[i] = v_t

        self.brain.train(x, y, batch_size=len_batch)

    def Results(self, size):
        return self.resultFile.Results(size)



ParamsOrderToCalibrate = [("kp", 0), ("kp", 1), ("kp", 2), 
                        ("kd", 0), ("kd", 1), ("kd", 2),
                        ("td", 0), ("td", 1), ("td", 2), 
                        ("ki", 0), ("ki", 1), ("ki", 2), 
                        ("ti", 0), ("ti", 1), ("ti", 2)]

class AlgoPID(AlgoBase):
    def __init__(self, num_state, num_action, configDict, train=True):
        super(AlgoPID, self).__init__(num_state, num_action, configDict)
        self.next_model(configDict)

        self.zTargetDist = configDict["zDistSuccess"]
        self.posZIdx = 0
        self.velZIdx = 1
        self.thrustIdx = 2

        self.doNothingAction = int(num_action / 2) + 1

        self.prevDist = {self.posZIdx : 0, self.velZIdx : 0, self.thrustIdx : 0}
        self.prevDistInit = False
        self.allIntegarls = {self.posZIdx : 0.0, self.velZIdx : 0.0, self.thrustIdx : 0.0}
        
        # constants
        self.Ku = {self.posZIdx : 1 / 2.8, self.velZIdx : 1 / 0.1, self.thrustIdx : 0.2}
        self.Tu = {self.posZIdx : 27, self.velZIdx : 18, self.thrustIdx : 20}

        self.pidParams = {"kp" : {0:0,1:0,2:0},
                            "kd" : {0:0,1:0,2:0},
                            "td" : {0:1,1:1,2:1},
                            "ki" : {0:0,1:0,2:0},
                            "ti" : {0:1,1:1,2:1},
                            "max": {0:float('inf'),1:float('inf'),2:float('inf')},
                            "min": {0:float('-inf'),1:float('-inf'),2:float('-inf')}}
        
        self.calibrate = "calibrate" in configDict and configDict["calibrate"]
        if self.calibrate:
            self.calibrateTable = CalibrateTable(configDict["directory"] + "/calibarateTable")
            self.finalCal = open(configDict["directory"] + "/finalParams.txt", "w+")
            self.currParamIdx = 0
            self.numRepeats = 200
            self.currValue = []
            self.prevValue = -1
            self.initStepSize = {"kp" : 0.1, "kd" : 0.1, "td" : 1, "ki" : 0.1, "ti" : 1}
            self.stepSize = self.initStepSize[ParamsOrderToCalibrate[self.currParamIdx][0]]
            self.minStepSize = 0.00001
            self.operator = operator.add

        if "constFile" in configDict:
            self.loadFromFile = configDict["directory"] + "/" + configDict["constFile"] + ".txt"
            self.load_const()
        else:
            self.loadFromFile = None
            #self.init_const()
        
        
        self.maxVal = 10
        self.duration = 50

        self.locs = []
        self.vels = []
        self.thrusts = []
        self.actions = []
        self.errorVel = []
        self.errorZ = []
        self.velI = []

        self.pidParams["kp"][self.velZIdx] = 0.9
        self.pidParams["ki"][self.velZIdx] = 0.01
        self.pidParams["ti"][self.velZIdx] = 0.05

        self.pidParams["max"][self.velZIdx] = 0.5
        self.pidParams["min"][self.velZIdx] = -0.5

        self.pidParams["kp"][self.posZIdx] = 0.2
        self.pidParams["max"][self.posZIdx] = 1
        self.pidParams["min"][self.posZIdx] = -1

        open(configDict["directory"] + "/params.txt", "w+").write(str(self.pidParams))

    def load_const(self):
        self.pidParams = eval(open(self.loadFromFile, "r+").read())

    
    def init_const(self):
        
        for key, Ku in self.Ku.items():
            Tu = self.Tu[key]
            if Ku != 0:
                self.pidParams["kp"][key] = 0.8 * Ku
                self.pidParams["kd"][key] = Ku * Tu / 10
                self.pidParams["td"][key] = float(Tu) / 8
                self.pidParams["ki"][key] = 0.0
                self.pidParams["ti"][key] = 0.0

    def act(self, state):
        prevD = self.prevDist[self.posZIdx]
        prevV = self.prevDist[self.velZIdx]
        prevT = self.prevDist[self.thrustIdx]
        
        targetVel, errZ = self.pid(state, self.posZIdx, 0)
        required_thrust, errVel = self.pid(state, self.velZIdx, targetVel)

        #print "thrust =", thrust
        output = max(min(0.9, required_thrust), 0.1) 
        # output /= self.maxVal
        
        self.actions.append(output)
        # output += 0.5

        if not self.prevDistInit:
            self.prevDistInit = True
            return 0.5 
    
        # output = max(min(output, 1.0), 0.0)  

        self.errorVel.append(errVel)
        self.errorZ.append(errZ)
        self.locs.append(state[self.posZIdx])
        self.vels.append(state[self.velZIdx])
        self.thrusts.append(state[self.thrustIdx])
        self.velI.append(self.allIntegarls[self.velZIdx])
        return output

    def end_episode(self, r, score, steps, realR):
        self.prevDistInit = False
        saved = False
        if not self.calibrate:
            saved = self.resultFile.end_run(r, score, steps, realR)

        self.prevDist = {self.posZIdx : 0, self.velZIdx : 0, self.thrustIdx : 0}
        self.allIntegarls = {self.posZIdx : 0.0, self.velZIdx : 0.57, self.thrustIdx : 0.0}

        if self.loadFromFile != None:
            self.load_const()
        elif self.calibrate:
            self.currValue.append(r)
            if len(self.currValue) == self.numRepeats:
                paramsList = self.ParamsToList()
                value = np.average(self.currValue)
                valueStd = np.std(self.currValue)
                self.currValue = []
                value = self.calibrateTable.insert(paramsList, value)
                
                newParam = False
                if self.prevValue > value:
                    if self.operator == operator.add:
                        self.operator = operator.sub
                    else:
                        self.operator = operator.add

                        self.stepSize *= 0.1
                        if self.stepSize < self.minStepSize:
                            newParam = True
                
                self.prevValue = value

                if newParam:
                    self.calibrateTable.Save()
                    saved = True
                    print "\n\nfinish param calibration", ParamsOrderToCalibrate[self.currParamIdx], "avg = " , value
                    self.prevValue = -1

                    self.currParamIdx = (self.currParamIdx + 1) % len(ParamsOrderToCalibrate)
                    self.stepSize = self.initStepSize[ParamsOrderToCalibrate[self.currParamIdx][0]]
                    print "new calibration =", ParamsOrderToCalibrate[self.currParamIdx], "\n"
                    if self.currParamIdx == 0:
                        self.finalCal.write(str(paramsList) + "\n")
                        self.finalCal.flush()
                else:
                    currParam = ParamsOrderToCalibrate[self.currParamIdx]
                    s = "param val = " + str(self.pidParams[currParam[0]][currParam[1]]) + ", avg = " + str(value)
                    self.pidParams[currParam[0]][currParam[1]] = self.operator(self.pidParams[currParam[0]][currParam[1]], self.stepSize)
                    print s, ", newVal =", self.pidParams[currParam[0]][currParam[1]]

        
        fig = plt.figure(figsize=(19.0, 11.0))

        plt.plot(np.arange(len(self.locs)) * 50, self.locs)
        plt.plot(np.arange(len(self.vels)) * 50, self.vels)
        plt.plot(np.arange(len(self.thrusts)) * 50, self.thrusts)
        plt.plot(np.arange(len(self.actions)) * 50, self.actions)
        plt.plot(np.arange(len(self.errorVel)) * 50, self.errorVel)
        plt.plot(np.arange(len(self.errorZ)) * 50, self.errorZ)
        plt.plot(np.arange(len(self.velI)) * 50, self.velI)
        plt.legend(["loc", "vel", "thrust", "required thrust", "errorVel", "errorZ", "integral"])
        plt.yticks(np.arange(-1.0,1.1,0.1))
    
        
        fig.savefig("./PID2/" + str(self.episodes) + ".png")
        plt.show()
        
        self.locs = []
        self.vels = []
        self.thrusts = []
        self.actions = []
        self.errorVel = []
        self.errorZ = []
        self.episodes += 1
        return saved, ""

    def ParamsToList(self):
        l = []
        keysSorted = sorted(list(self.pidParams.keys()))
        for k in keysSorted:
            keysSorted2 = sorted(list(self.pidParams[k].keys()))
            for k2 in keysSorted2:
                l.append(self.pidParams[k][k2])
        
        return l

            
    def pid(self, state, key, target):

        dist = target - state[key]
        maxD = self.pidParams["max"][key]
        minD = self.pidParams["min"][key]
        dist = max(min(dist, maxD), minD)
        
        p = self.pidParams["kp"][key] * dist
        d = self.pidParams["kd"][key] * (dist - self.prevDist[key]) / self.pidParams["td"][key]
        
        pid = p + d + self.allIntegarls[key]
        self.allIntegarls[key] += self.pidParams["ki"][key] * dist * self.pidParams["ti"][key]
        self.allIntegarls[key] = np.clip(self.allIntegarls[key], 0.49, 0.65)

        self.prevDist[key] = dist
        return pid, dist

    def real_action(self):
        return True


class CalibrateTable:
    def __init__(self, fname):
        self.fname = fname
        self.cols = list(range(2))
        self.countIdx = 0
        self.valueIdx = 1
        self.table = pd.DataFrame(columns=self.cols, dtype=np.float)

        self.preciseness = 5

    def Load(self):
        # if file exist reaf table and read from table complete count key
        if os.path.isfile(self.fname + '.gz') and os.path.getsize(self.fname + '.gz') > 0:
            self.table = pd.read_pickle(self.fname + '.gz', compression='gzip')
            return True
        
        return False

    def Save(self):
        self.table.to_pickle(self.fname + '.gz', 'gzip') 

    def check_state_exist(self, state):
        if state not in self.table.index:
            # append new state filled with 0
            self.table = self.table.append(pd.Series([0] * len(self.cols), index=self.table.columns, name=state))
            return False
        
        return True

    def insert(self, params, value):
        allParams = [ round(p * (10 ** self.preciseness)) / (10 ** self.preciseness) for p in params]
        key = str(allParams)
        
        if self.check_state_exist(key):
            count = self.table.ix[key, self.countIdx] + 1
            orgVal = self.table.ix[key, self.valueIdx]
            value = orgVal + (value - orgVal) / count
        else:
            count = 1

        self.table.ix[key, self.valueIdx] = value
        self.table.ix[key, self.countIdx] = count

        return value