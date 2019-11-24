import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import math

from algo_decisionMakers import AlgoBase

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, num_states, num_actions, size, fname=""):
        self.fname = fname
        self.state = np.zeros([size, num_states], dtype=np.float32)
        self.state_next = np.zeros([size, num_states], dtype=np.float32)
        self.actions = np.zeros([size, num_actions], dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.terminal = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, s, a, r, s_, done):
        self.state[self.ptr] = s
        self.state_next[self.ptr] = s_
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.terminal[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {"s" : self.state[idxs],
                "s_" : self.state_next[idxs],
                "a" : self.actions[idxs],
                "r" : self.rewards[idxs],
                "d" : self.terminal[idxs]}
    
    def save(self):
        if self.fname != "":
            d = {"s" : self.state,
                "s_" : self.state_next,
                "a" : self.actions,
                "r" : self.rewards,
                "d" : self.terminal,
                "size" : self.size,
                "ptr" : self.ptr}
            
            pd.to_pickle(d, self.fname + ".gz", "gzip")
    
    def load(self):
        if self.fname != "" and os.path.isfile(self.fname + ".gz"):
            d = pd.read_pickle(self.fname + ".gz", "gzip")
            
            self.state = d["s"]
            self.state_next = d["s_"]
            self.actions = d["a"]
            self.rewards = d["r"]
            self.terminal = d["d"]
            self.size = d["size"]
            self.ptr = d["ptr"]

    def load_all(self):
        dAll = []
        if self.fname != "" and os.path.isfile(self.fname + ".gz"):
            dAll.append(pd.read_pickle(self.fname + ".gz", "gzip"))

        currIdx = 0
        currFname = self.fname + str(currIdx)
        while os.path.isfile(currFname + ".gz"):
            dAll.append(pd.read_pickle(currFname + ".gz", "gzip"))
            currIdx += 1
            currFname = self.fname + str(currIdx)

        dataDict = {"s" : [], "s_" : [], "a" : [], "r" : [], "d" : []}
        size = 0
        for d in dAll:
            for k in dataDict.keys():
                dataDict[k] += list(d[k])
            
            size += d["size"]
        
        self.state = np.array(dataDict["s"])
        self.state_next = np.array(dataDict["s_"])
        self.actions = np.array(dataDict["a"])
        self.rewards = np.array(dataDict["r"])
        self.terminal = np.array(dataDict["d"])
        self.size = size
        self.ptr = size
        self.max_size = size

class AgentTD3(AlgoBase):
    def __init__(self, num_state, num_actions, configDict, train=True, gamma=0.9, tau=0.96, actor_learning_rate=5e-6, critic_learning_rate=2e-4, batch=64, num_updates=64):
        num_state = 2
        super(AgentTD3, self).__init__(num_state, num_actions, configDict, createResults=False, numToWrite=10)
        self.modelFName = ""
        self.directory = configDict["directory"]
        # Params
        self.num_state = num_state
        self.num_actions = num_actions
        self.act_limit = 1.0 if "action_lim" not in configDict else configDict["action_lim"]
        self.act_offset = 0.0 if "action_offset" not in configDict else configDict["action_offset"]
        self.act_max = self.act_limit + self.act_offset
        self.act_min = -self.act_limit + self.act_offset

        hidden_size = (256,256,) if "hidden_size" not in configDict else configDict["hidden_size"]

        self.batch_size = batch
        self.max_memory_size=500000
        self.random_episodes = 30
        self.num_updates = num_updates
        self.updateSum = 0

        # data struct
        self.noise = 0.1 * self.act_limit
        targetNoise = 0.2 * self.act_limit
        targetNoiseClip = 0.5 * self.act_limit

        self.replay_buffer = ReplayBuffer(num_state, num_actions, self.max_memory_size, configDict["directory"] + "/history")  
        self.currFileNum = 0   
        with tf.variable_scope("ddpg"):
            # inputs
            self.states = tf.placeholder(tf.float32, shape=[None, self.num_state], name="state")
            self.states_next = tf.placeholder(tf.float32, shape=[None, self.num_state], name="state_next")
            self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name="actions")
            self.norm_actions = (self.actions - self.act_offset) / self.act_limit
            
            self.rewards = tf.placeholder(tf.float32, shape=[None,1], name="rewards")
            self.terminal = tf.placeholder(tf.float32, shape=[None,1], name="terminal")

            # Networks  
            with tf.variable_scope("main"):
                self.actor, self.critic1, self.critic2, self.c1_a = self.actor_critic(self.states, hidden_size)

            # Target policy network
            with tf.variable_scope("target"):
                self.actor_target, _, _, _ = self.actor_critic(self.states_next, hidden_size)

            # Target Q networks
            with tf.variable_scope('target', reuse=True):
                # Target policy smoothing, by adding clipped noise to target actions
                epsilon = tf.random_normal(tf.shape(self.actor_target), stddev=normalizedNoise)
                epsilon = tf.clip_by_value(epsilon, -targetNoiseClip, targetNoiseClip)
                aWithNoise = self.actor_target + epsilon
                a2 = tf.clip_by_value(a2, -act_limit, act_limit)

                # Target Q-values, using action from target policy
                _, self.critic1_target, self.critic2_target, _ = actor_critic(x2_ph, a2, **ac_kwargs)
            
            # Bellman equation
            self.bellman = tf.stop_gradient(self.rewards + gamma*(1-self.terminal)*self.c_a_target)   

            # DDPG losses
            # actor = -avg(critic(s, actor(s)))
            # critic = mse(critic(s,a) - bellman)
            self.loss_actor = -tf.reduce_mean(self.c_a)
            self.loss_critic = tf.reduce_mean((self.critic-self.bellman)**2)

            # Separate train ops for pi, q
            
            actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_learning_rate)
            critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_learning_rate)
            
            # Polyak averaging for target variables
            self.target_update_op = tf.group([tf.assign(v_targ, tau*v_targ + (1-tau)*v_main)
                                    for v_main, v_targ in zip(self.get_vars('main'), self.get_vars('target'))])

            # Initializing targets to match main variables
            self.target_init = tf.group([tf.assign(v_targ, v_main)
                                    for v_main, v_targ in zip(self.get_vars('main'), self.get_vars('target'))])

            self.train_actor_op = actor_optimizer.minimize(self.loss_actor, var_list=self.get_vars('main/actor'))
            self.train_critic_op = critic_optimizer.minimize(self.loss_critic, var_list=self.get_vars('main/critic'))
            
        self.next_model(configDict)
        
        self.sess = tf.Session()
        self.init_params()
        
        if not os.path.exists(configDict["directory"] + "/figs"):
            os.makedirs(configDict["directory"] + "/figs")

    def init_params(self, resetHist=False):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init)
        self.updateSum = 0
        if resetHist:
            self.replay_buffer.size = 0
            self.replay_buffer.ptr = 0

    def real_action(self):
        return True
    
    def next_model(self, configDict, load=False):
        if self.modelFName != "":
            self.save_model(self.modelFName)
        super(AgentTD3, self).next_model(configDict, load)
        self.modelFName = configDict["directory"] + "/ddpg_"
        modelName = str(configDict["zDistSuccess"]) + "_" + str(configDict["restartType"])  + "_" + str(configDict["initZOptions"])
        modelName = modelName.replace(" ","")
        self.modelFName +=  modelName
        self.figFName = configDict["directory"] + "/figs/" + modelName
        self.figFName.replace(".",",")

    def get_vars(self, scope):
        nnVars = [x for x in tf.global_variables() if scope in x.name]
        return sorted(nnVars, key=lambda v: v.name)

    def load_model(self, name, scope="ddpg"):
        modelParams = pd.read_pickle(name + ".gz", "gzip")
        
        nnVars = self.get_vars(scope)

        copy_ops = []
        for v in range(len(nnVars)):
            tgtName = nnVars[v].name
            if tgtName in modelParams:
                op = nnVars[v].assign(modelParams[nnVars[v].name])
                copy_ops.append(op)
            else:
                print "didnt found var name =", tgtName, "!!"

        self.sess.run(copy_ops) 

    def save_model(self, name, scope="ddpg"):  
        nnVars = self.get_vars(scope)

        modelParams = {}
        varName = []
        for v in range(len(nnVars)):
            modelParams[nnVars[v].name] = nnVars[v].eval(session = self.sess)
            
        pd.to_pickle(modelParams, name + ".gz", "gzip")

    def load(self):
        if os.path.isfile(self.modelFName + ".gz"):
            print "loaded ->", self.modelFName
            self.load_model(self.modelFName)
            loaded = True
        else:
            print self.modelFName, "not exist"
            loaded = False
        
        self.resultFile.Load()
        self.replay_buffer.load()
        if loaded:
            self.episodes = self.resultFile.NumRuns()
            self.random_episodes -= self.episodes
        print "model num episdoes =", self.episodes, "numRuns =", self.resultFile.NumRuns()
        
    def actor_critic(self, s, hidden_size):
        with tf.variable_scope("actor"):
            actor = self.act_offset + self.act_limit * self.net(s,hidden_size + (self.num_actions,))
            
        with tf.variable_scope("critic1"):
            critic1_input = tf.concat([s, self.norm_actions], axis=-1) 
            critic1 = self.net(critic1_input, hidden_size + (1,), output_activation=None)

        with tf.variable_scope("critic2"):
            critic2_input = tf.concat([s, self.norm_actions], axis=-1) 
            critic2 = self.net(critic2_input, hidden_size + (1,), output_activation=None)

        with tf.variable_scope("critic1", reuse=True): 
            ca_input = tf.concat([s, actor], axis=-1) 
            c_a = self.net(ca_input, hidden_size + (1,), output_activation=None)
        
        return actor, critic1, critic2, c_a

    def net(self, inputLayer, hidden_sizes=(32,), activation=tf.nn.relu, output_activation=tf.tanh):  
        currI = inputLayer
        for h in hidden_sizes[:-1]:
            currI = tf.layers.dense(currI, units=h, activation=activation)
        return tf.layers.dense(currI, units=hidden_sizes[-1], activation=output_activation)

    def act(self, state, target=False, noise=True):
        state = state[:self.num_state]
        if self.random_episodes > 0 and target == False:
            return self.random_action()
        
        actor = self.actor
        action = actor.eval({self.states: state.reshape(1, self.num_state)}, session=self.sess).reshape(self.num_actions)
        action += self.noise * np.random.randn(self.num_actions) if noise else 0

        return np.clip(action, self.act_min, self.act_max) 

    def random_action(self):
        sig = self.act_limit / 2
        action = sig * np.random.randn(self.num_actions) + self.act_offset
        return np.clip(action, self.act_min, self.act_max) 
    def state_value(self, s):
        return self.c_a_target.eval({self.states_next: s.reshape(1, self.num_state)}, session=self.sess)
        
    def observe(self, s,a,r,s_,done):
        s = s[:self.num_state]
        s_ = s_[:self.num_state]
        self.replay_buffer.store(s,a,r,s_,done)

    def update(self, num_updates=1):       
        for _ in range(num_updates):
            batch = self.replay_buffer.sample(self.batch_size)

            feed_dict = {self.states: batch["s"],
                        self.states_next: batch["s_"],
                        self.actions: batch["a"],
                        self.rewards: batch["r"].reshape(self.batch_size,1),
                        self.terminal: batch["d"].reshape(self.batch_size,1)}

            # actor critic learning (seperate learning)
            lossCritic, _ = self.sess.run([self.loss_critic, self.train_critic_op], feed_dict)
            lossActor, _ = self.sess.run([self.loss_actor, self.train_actor_op], feed_dict)

            # update target
            self.sess.run(self.target_update_op)

        self.updateSum += num_updates

    def end_episode(self, r, score, steps, realR):
        if self.episodes == 0:
            self.save_fig(self.figFName + "_init")

        num_updates = min(self.num_updates, int(self.replay_buffer.size / self.batch_size))
        self.update(num_updates)
        self.episodes += 1
        self.random_episodes -= 1
        saveModel = self.resultFile.end_run(r, score, steps, realR)
        if saveModel:
            print "\tcreating figures and saving model..."
            self.save_model(self.modelFName)
            self.replay_buffer.save()
            if self.replay_buffer.size + 2000 > self.max_memory_size:
                self.replay_buffer = ReplayBuffer(self.num_state, self.num_actions, self.max_memory_size, 
                                                self.directory + "/history" + str(self.currFileNum))  
                self.currFileNum += 1   
            self.save_fig(self.figFName + str(self.episodes))

        return saveModel
    
    def save_fig(self, name, z=None, v=None, vals=None):
        if z == None:
            z, v, vals = self.create_fig_data()
        
        
        fig = plt.figure(figsize=(19.0, 11.0))
        ixPlot = 1
        for key, mat in vals.items():
            ax = plt.subplot(2,2,ixPlot)

            if key.find("action") >= 0:
                pos = plt.imshow(mat, cmap='bwr')
                plt.clim(self.act_offset - self.act_limit, self.act_offset + self.act_limit)
            else:
                pos = plt.imshow(mat)
        
            zt = np.arange(0,len(z),10)
            labels = np.round(z[zt] * 100) / 100    
            ax.set_yticks(zt)
            ax.set_yticklabels(labels)
            plt.ylabel("z")

            vt = np.arange(0,len(v),10)
            labels = np.round(v[vt] * 100) / 100   
            ax.set_xticks(vt)
            ax.set_xticklabels(labels)
            plt.xlabel("velo")
            
            fig.colorbar(pos)
        
            plt.title(key + " value")
            ixPlot += 1
        
        fig.suptitle("num updates = " + str(self.updateSum), fontsize=20)
        fig.savefig(name + ".png")
        plt.close()
        return z, v, vals
    
    def create_fig_ax(self):
        z = np.arange(5.0,-5.0,-0.1)
        v = np.arange(-2,2,0.1)
        return z, v
        
    def create_fig_data(self):
        z, v = self.create_fig_ax()
        vals = {"action" : np.zeros((len(z), len(v))), "state" : np.zeros((len(z), len(v))), 
                "action_tgt" : np.zeros((len(z), len(v))), "state_tgt" : np.zeros((len(z), len(v)))}
        
        for zi in range(len(z)):
            for vi in range(len(v)):
                s = np.array([z[zi], v[vi]]).reshape(1, self.num_state)
                aVal, sVal = self.sess.run([self.actor, self.c_a], {self.states: s})
                aValTgt, sValTgt = self.sess.run([self.actor_target, self.c_a_target], {self.states_next: s})

                vals["action"][zi, vi] = aVal.squeeze()
                vals["state"][zi, vi] = sVal.squeeze()
                vals["action_tgt"][zi, vi] = aValTgt.squeeze()
                vals["state_tgt"][zi, vi] = sValTgt.squeeze()

        return z, v, vals
    

def getModel(loadAllhist, dirHist, gamma, tau, lr_pi, lr_q, batch):
    configDict = eval(open(dirHist+"/config.txt", "r+").read())
    configDict["directory"] = dirHist
    configDict["zDistSuccess"] = ""
    configDict["restartType"] = ""
    configDict["initZOptions"] = ""
    self = AgentTD3(2,1,configDict,  gamma=gamma, tau=tau, actor_learning_rate=lr_pi, critic_learning_rate=lr_q, batch=batch)
    if loadAllhist:
        self.replay_buffer.load_all()
    else:
        self.replay_buffer.load()
    print "hist size =", self.replay_buffer.size
    return self

def im(hist, maxZ=5.0, maxV=3.0):
    zBase = np.arange(maxZ, -maxZ, -0.1)
    vBase = np.arange(maxV, -maxV, -0.1)

    mat = np.zeros((len(zBase), len(vBase)))
    mat.fill(np.nan)
    for i in range(hist.size):
        sz = hist.state[i][0]
        sv = hist.state[i][1]
        if abs(sz) < maxZ and abs(sv) < maxV:
            iz = np.argmin(abs(zBase - sz))
            iv = np.argmin(abs(vBase - sv))
            if np.isnan(mat[iz, iv]):
                mat[iz, iv] = hist.rewards[i]
            else:
                mat[iz, iv] +=  0.1 * (hist.rewards[i] - mat[iz, iv])

    plt.imshow(mat)
    plt.show()
    return mat


def learnOffline(model, allResults, path, numRepetitions, numEpisodes, numUpdates=1, restart=False, runByOrder=True):
    
    replay_buffer = model.replay_buffer
    model.replay_buffer = ReplayBuffer(2,1,50000)
    done = replay_buffer.terminal.nonzero()[0]
    start = np.array([0] + list(done[:-1] + 1))

    
    maxNumEpisodes = int(min(len(start), numEpisodes))

    print "numep =", maxNumEpisodes
    collectDataJumps = 10
    
    if restart or -1 not in allResults:
        base = list(range(0,maxNumEpisodes+1, collectDataJumps))
        base[0] = -1
        for b in base:
            allResults[b] = {"action" : [], "state" : [], "action_tgt" : [], "state_tgt" : []}
        startRep = 0
    else:
        startRep = len(allResults[-1]["action"])
        print "start rep from", startRep


    for rep in range(startRep, numRepetitions):
        print "train #", rep
        results = {}
        folder = path + str(rep)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        
        folder += "/"
        model.init_params(resetHist=True)
        z, v , vals = model.save_fig(folder + "-1")
        results[-1] = {}
        for key in vals.keys():
            results[-1][key] = vals[key]
        
        for episode in range(maxNumEpisodes):
            if runByOrder:
                curr = start[episode]
            else:
                idx = np.random.choice(start)
                curr = start[idx]

            while not replay_buffer.terminal[curr]: 
                model.observe(replay_buffer.state[curr],
                            replay_buffer.actions[curr],
                            replay_buffer.rewards[curr],
                            replay_buffer.state_next[curr],
                            replay_buffer.terminal[curr])
            
                curr += 1

            currUpdates = min(int(model.replay_buffer.size / model.batch_size), numUpdates)
            model.update(currUpdates)
                
            if (episode + 1) % collectDataJumps == 0:
                _,_, vals = model.save_fig(folder + str(episode + 1))

                results[episode + 1] = {}
                for key in vals.keys():
                    results[episode + 1][key] = vals[key]
                    

        for ep, data in results.items():
            for key, mat in data.items():
                allResults[ep][key].append(mat) 
        
if __name__ == "__main__":
    from absl import flags
    import sys

    flags.DEFINE_string("learn", "sim", "")
    flags.DEFINE_float("tau", 0.995, "")
    flags.DEFINE_float("gamma", 0.9, "")
    
    flags.DEFINE_float("lr_pi", 1e-4, "")
    flags.DEFINE_float("lr_q", 1e-3, "")
    
    flags.DEFINE_integer("numTrains", 20, "")
    flags.DEFINE_float("numEpisodes", float('inf'), "")
    flags.DEFINE_string("directory", "auto", "")
    flags.DEFINE_string("add2Dir", "", "")
    flags.DEFINE_bool("runByOrder", True, "")
    flags.DEFINE_integer("numUpdates", 1, "")
    flags.DEFINE_integer("batchSize", 128, "")


    flags.FLAGS(sys.argv)

    learn = flags.FLAGS.learn

    dirName = "Random"
    tau = flags.FLAGS.tau 
    gamma = flags.FLAGS.gamma
    
    if flags.FLAGS.directory == "auto":
        order = "" if flags.FLAGS.runByOrder else "_shuffled"
        if flags.FLAGS.lr_pi != 1e-4 or flags.FLAGS.lr_q != 1e-3:
            addlr = "_pi=" + str(flags.FLAGS.lr_pi) + "_q=" + str(flags.FLAGS.lr_q)
        else:
            addlr = ""
        if flags.FLAGS.numUpdates != 1:
            numUpdates = "_numUpdates=" + str(int(flags.FLAGS.numUpdates))
        else:
            numUpdates = ""

        if flags.FLAGS.numEpisodes != 100:
            addNe = "_numEp=" + str(int(flags.FLAGS.numEpisodes))
        else:
            addNe = ""

        if flags.FLAGS.batchSize != 128:
            addBatch = "_batch=" + str(int(flags.FLAGS.batchSize))
        else:
            addBatch = ""

        
        f = "./" + dirName + "/offline_g=" + str(gamma) + "_t=" + str(tau) + addlr + addNe + addBatch + numUpdates + order
    else:
        f = "./" + dirName + "/" + flags.FLAGS.directory

    if flags.FLAGS.add2Dir != "":
        f += "_" + flags.FLAGS.add2Dir
    f += "/"
    
    if not os.path.isdir(f):
        os.makedirs(f)

    print "save in dir =", f
    if os.path.isfile(f + "results.gz"):
        allResults = pd.read_pickle(f + "results.gz", 'gzip')
    else:
        allResults = {}

    model = getModel(loadAllhist=False, dirHist=dirName, gamma=gamma, tau=tau, 
                lr_pi=flags.FLAGS.lr_pi, lr_q=flags.FLAGS.lr_q, batch=flags.FLAGS.batchSize)
    
    if learn == "sim":   
        learnOffline(model, allResults, f, flags.FLAGS.numTrains, numUpdates=flags.FLAGS.numUpdates,
                    numEpisodes=flags.FLAGS.numEpisodes, runByOrder=flags.FLAGS.runByOrder)
        pd.to_pickle(allResults, f + "results.gz", 'gzip')


    folder = f + "all/"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    
    z, v = model.create_fig_ax()

    for episode, allVals in allResults.items():
        
        fig = plt.figure(figsize=(19.0, 11.0))
        ixPlot = 1
        for key, allData in allVals.items():
            for dataType in ["avg", "std"]:
                mat = np.array(allData)
                data = np.average(mat, axis=0) if dataType == "avg" else np.std(mat, axis=0)
                ax = plt.subplot(3,3,ixPlot)

                
                if key.find("action") >= 0 and dataType == "avg":
                    pos = plt.imshow(data, cmap='bwr')
                    plt.clim(0.49, 0.65)
                else:
                    pos = plt.imshow(data)
                    plt.clim(np.min(data), np.max(data))
                
        
                zt = np.arange(0,len(z),10)
                labels = np.round(z[zt] * 100) / 100    
                ax.set_yticks(zt)
                ax.set_yticklabels(labels)
                plt.ylabel("z")

                vt = np.arange(0,len(v),10)
                labels = np.round(v[vt] * 100) / 100   
                ax.set_xticks(vt)
                ax.set_xticklabels(labels)
                plt.xlabel("velo")
                
                fig.colorbar(pos)
            
                plt.title(dataType + " " + key + " value")
                ixPlot += 1
        
        fig.suptitle("num episodes = " + str(episode), fontsize=20)
        fig.savefig(folder + str(episode) + ".png")          
    
    maxEpisode = max(allResults.keys())
    for key, allData in allResults[maxEpisode].items():
        fig = plt.figure(figsize=(19.0, 11.0))
        numPlots1Ax = int(math.ceil(math.sqrt(len(allData))))
        for i in range(len(allData)):
            data = allData[i]
            ax = plt.subplot(numPlots1Ax,numPlots1Ax,i+1)

            if key.find("action") >= 0:
                pos = plt.imshow(data, cmap='bwr')
                plt.clim(0.49, 0.65)
            else:
                pos = plt.imshow(data)
                plt.clim(np.min(data), np.max(data))
            
    
            zt = np.arange(0,len(z),10)
            labels = np.round(z[zt] * 100) / 100    
            ax.set_yticks(zt)
            ax.set_yticklabels(labels)
            plt.ylabel("z")

            vt = np.arange(0,len(v),10)
            labels = np.round(v[vt] * 100) / 100   
            ax.set_xticks(vt)
            ax.set_xticklabels(labels)
            plt.xlabel("velo")
            
            fig.colorbar(pos)
        
    
        fig.suptitle(str(maxEpisode) + " episodes " + key + " value", fontsize=20)
        fig.savefig(folder + "all_" + key + ".png")  