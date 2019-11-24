import numpy as np
import pandas as pd
import tensorflow as tf
import os   
import math

from algo_decisionMakers import AlgoBase

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

MODEL_NAME = "DTM"

class ReplayBuffer:
    def __init__(self, stateSize, outputSize, idxCurrentState, size, fname=""):
        self.fname = fname
        
        self.state = np.zeros([size, stateSize], dtype=np.float32)
        self.state_diff = np.zeros([size, outputSize], dtype=np.float32)
        self.state_next = np.zeros([size, outputSize], dtype=np.float32)
        self.idxCurrentState = idxCurrentState
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, s, s_):    
        self.state[self.ptr,:] = s
        self.state_next[self.ptr,:] = s_
        # calculte sDiff
        self.state_diff[self.ptr,:] = s_ - s[self.idxCurrentState]

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {"s" : self.state[idxs,:],
                "s_" : self.state_next[idxs,:], 
                "labels" : self.state_diff[idxs,:]}
    
    def save(self):
        if self.fname != "":
            d = {"s" : self.state,
                "s_" : self.state_next,
                "labels" : self.state_diff,
                "size" : self.size,
                "ptr" : self.ptr}
            
            print "saving dtm hist in", self.fname
            pd.to_pickle(d, self.fname + ".gz", "gzip")
    
    def load(self):
        if self.fname != "" and os.path.isfile(self.fname + ".gz"):
            d = pd.read_pickle(self.fname + ".gz", "gzip")
            
            self.state = d["s"]
            self.state_diff = d["labels"]
            self.state_next = d["s_"]
            self.size = d["size"]
            self.ptr = d["ptr"]       

class DTM:
    def __init__(self, num_state, num_actions, configDict, learning_rate=1e-6, batch=64, num_updates=64, directory="DTM", modelName=MODEL_NAME):
        self.state_hist_size = configDict["state_hist_size"] if "state_hist_size" in configDict else 0

        self.directory = directory
        self.modelFName = self.directory + "/" + modelName
        # Params

        self.num_state = num_state
        self.num_actions = num_actions
        self.stateSize = (num_state + num_actions) * self.state_hist_size
        self.currStateIdx = np.arange(self.num_state)
        
        self.InitHist()
        self.state_norm = np.array([1.0, 1.0, 1.0])
        
        hidden_size = (256,256,) if "hidden_size" not in configDict else configDict["hidden_size"]

        self.batch_size = batch
        self.max_memory_size=500000
        self.updateFreq = 10
        self.num_updates = num_updates * self.updateFreq
        self.updateSum = 0
        self.episodes = 0

        # data struct 
        # hard coded states idx first members in StateInput
        self.replay_buffer = ReplayBuffer(self.stateSize, self.num_state, self.currStateIdx, self.max_memory_size, 
                                            self.directory + "/" + MODEL_NAME +"history") 
        self.currFileNum = 0   

        with tf.variable_scope(self.modelFName):
            # inputs
            self.states = tf.placeholder(tf.float32, shape=[None, self.stateSize], name="state")
            self.labels = tf.placeholder(tf.float32, shape=[None, self.num_state], name="state_next")
            
            # Networks  
            self.predictions = self.model(self.states, hidden_size + (self.num_state,))

            self.losses = ((self.predictions - self.labels) / self.state_norm) ** 2
            self.loss = tf.reduce_mean(self.losses)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                
            self.train = optimizer.minimize(self.loss)

    def InitSession(self, sess):
        self.sess = sess
    
    def InitHist(self):
        self.actionsHist = [np.zeros(self.num_actions) for _ in range(self.state_hist_size)]
        self.statesHist = [np.zeros(self.num_state) for _ in range(self.state_hist_size)]
        
    def get_vars(self, scope):
        realScope = self.directory + "/" + scope
        nnVars = [x for x in tf.global_variables() if realScope in x.name]
        return sorted(nnVars, key=lambda v: v.name)

    def load_model(self, name, copy=False, scope=MODEL_NAME):
        modelParams = pd.read_pickle(name + ".gz", "gzip")
        
        nnVars = self.get_vars(scope)
        if copy:
            self.match_keys(nnVars, modelParams)

        copy_ops = []
        for v in range(len(nnVars)):
            tgtName = nnVars[v].name
            if tgtName in modelParams:
                op = nnVars[v].assign(modelParams[nnVars[v].name])
                copy_ops.append(op)
            else:
                print "didnt found var name =", tgtName, "!!"

        self.sess.run(copy_ops) 
    
    def match_keys(self, tgtVars, params):
        tgtList = [k.name for k in tgtVars]
        srcList = list(params.keys())

        tgtList.sort(key=lambda x : x[::-1])
        srcList.sort(key=lambda x : x[::-1])
        
        
        for i in range(len(tgtList)):
            params[tgtList[i]] = params.pop(srcList[i])

    def save_model(self, name, scope=""):  
        nnVars = self.get_vars(scope)

        modelParams = {}
        varName = []
        for v in range(len(nnVars)):
            modelParams[nnVars[v].name] = nnVars[v].eval(session = self.sess)
            
        pd.to_pickle(modelParams, name + ".gz", "gzip")

    def load(self, copy=False):
        if os.path.isfile(self.modelFName + ".gz"):
            print "loaded ->", self.modelFName
            self.load_model(self.modelFName, copy)
            loaded = True
        else:
            print self.modelFName, "not exist"
            loaded = False
        
        self.replay_buffer.load()
        if loaded:
            print "loaded model estimation, hist size =", self.replay_buffer.size
        
    def model(self, inputLayer, hidden_sizes, activation=tf.nn.relu, output_activation=tf.nn.tanh):  
        currI = inputLayer
        for h in hidden_sizes[:-1]:
            currI = tf.layers.dense(currI, units=h, activation=activation)

        output = tf.layers.dense(currI, units=hidden_sizes[-1], activation=output_activation)

        return output

    def get_state(self):
        s = np.array(self.statesHist)[:,:self.num_state].flatten()
        a = np.array(self.actionsHist).flatten()
        return np.concatenate((s, a))
        
    def observe(self, s, a, s_, done):
        self.actionsHist.pop(-1)
        self.actionsHist.insert(0, a)
        
        self.statesHist.pop(-1)
        self.statesHist.insert(0, s) 

        s = self.get_state()

        self.replay_buffer.store(s, s_)
        if done:
            self.InitHist()

    def update(self, num_updates=1):       
        for _ in range(num_updates):
            batch = self.replay_buffer.sample(self.batch_size)

            feed_dict = {self.states: batch["s"],
                        self.labels: batch["labels"]}

            loss, _ = self.sess.run([self.loss, self.train], feed_dict)

        self.updateSum += num_updates

    def end_episode(self):
        self.episodes += 1
        if (self.episodes) % self.updateFreq == 0:
            num_updates = min(self.num_updates, int(self.replay_buffer.size / self.batch_size))
            self.update(num_updates)
            
            self.save_model(self.modelFName)
            self.replay_buffer.save()
            
            if self.replay_buffer.size + 2000 > self.max_memory_size:
                # hard coded states idx first members in StateInput
                self.replay_buffer = ReplayBuffer(self.stateSize, self.num_state, self.currStateIdx, self.max_memory_size, 
                                                    self.directory + "/" + MODEL_NAME +"history" + str(self.currFileNum))  
                self.currFileNum += 1   

    def calc_loss(self, num_check=100):
        """calc losses"""
        allLoss = []
        for _ in range(num_check):
            batch = self.replay_buffer.sample(self.batch_size)

            feed_dict = {self.states: batch["s"],
                        self.labels: batch["labels"]}

            loss = self.sess.run(self.loss, feed_dict)
            allLoss.append(loss)
        
        return np.average(allLoss)
    
    def calc_all_losses(self, num_check=100):
        """calc losses including individual losses"""
        allLosses = []
        for _ in range(num_check):
            batch = self.replay_buffer.sample(self.batch_size)

            feed_dict = {self.states: batch["s"], self.labels: batch["labels"]}

            loss, losses = self.sess.run([self.loss, self.losses], feed_dict)
            losses = np.average(losses, axis=0)
            loss = [loss]
            allLosses.append(np.concatenate((loss, losses)))
                        
        
        return np.average(allLosses, axis=0)

    