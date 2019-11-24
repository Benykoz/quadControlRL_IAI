#!/usr/bin/env python
# An implementation of UAV DRL

# Func:
# 1) kepp the pos1 fixed in 6+-0.3  D!
#
# Implementation:
# 1) Work with player_test.py   D!
#
# Subscribe: game(Environment) status
# Publish: action: only sent when game status is received
#
# author: bingbing li 07.02.2018

import rospy
from drl_uav.msg import Num
from drl_uav.msg import Input_Game
from drl_uav.msg import Output_Game

# get UAV status:
from geometry_msgs.msg import PoseStamped	# UAV pos status
from geometry_msgs.msg import TwistStamped	# UAV vel status
from drl_uav.msg import Restart_Finished # UAV restart finished
from drl_uav.msg import AttControlRunning    # UAV att_control running: ready for Memory::observe().
from drl_uav.msg import AttitudeTarget       # UAV att setpoint(thrust is used)

import numpy as np
import pandas as pd
import random, math, gym
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import tensorflow as tf

import os
import sys
import time
from algo_decisionMakers import AlgoDQN
from algo_decisionMakers import AlgoPID, AgentTest
from algo_ddpg import AgentDDPG

from algo_dtm import DTM

from utils_results import ResultFile

from absl import flags


# terminal CMD:
# source ~/src/DRL_node_ROS/devel/setup.bash 
# rosrun drl_uav talker.py

AGENTS = {"dqn" : AlgoDQN, "pid" : AlgoPID, "switch" : AlgoPID, "ddpg" : AgentDDPG}

flags.DEFINE_string("directory", "Heuristic", "model directory")
flags.DEFINE_string("directoriesPrefix", "", "model directory")

flags.DEFINE_bool("resetModel", False, "load or reset model")
flags.DEFINE_bool("resetDtm", False, "load or reset model")

flags.DEFINE_bool("trainEndless", True, "load or reset model")
flags.DEFINE_float("numEpisodes", float('inf'), "")
flags.DEFINE_integer("dtmHistSize", 4, "")


REWARD_FUNC = {} 
TERMINAL_FUNC = {} 
TRANSFER_TRANSITIONS = {}

ENV_RATE = 20

TARGET_Z = 20.0

MAX_VELOCITY_Z = 2.25

NORMAL_STATE = False
Z_MAX_DIST = 10.0
Z_NORMAL_VAL = 10.0

THRUST_EQUALIBRIUM = 0.57
THRUST_NORMAL = 0.4

NUM_STEPS_2_TERMINATE = 200 #float('inf')
NON_VALID_NUM_STEPS = NUM_STEPS_2_TERMINATE + 1000

PLAYING = 2
DONE = 3
REAL_RESTART = 4
SET_STARTING_LOC = 5

STATUS_STR = {DONE: "DONE!", PLAYING : "PLAYING!", REAL_RESTART: "REAL_RESTART!", SET_STARTING_LOC : "SET_STARTING_LOC"}

# get UAV status:
UAV_Vel = TwistStamped()
UAV_Pos = PoseStamped()
att_running = AttControlRunning()
UAV_Att_Setpoint = AttitudeTarget()

def UAV_pos_callback(data):
    global UAV_Pos  
    # Subscribing:
    # rospy.loginfo('Receive UAV Pos Status: %f %f %f', data.pose.position.x, data.pose.position.y, data.pose.position.z)
    UAV_Pos = data

def UAV_vel_callback(data):
    global UAV_Vel
    # Subscribing:
    # rospy.loginfo('Receive UAV Vel Status: %f %f %f', data.twist.linear.x, data.twist.linear.y, data.twist.linear.z)  
    UAV_Vel = data

def restart_finished_callback(data):
    # Subscribing:
    # rospy.loginfo('UAV restart finished: %d', data.finished)
    pass


def att_running_callback(data):
    global att_running
    # Subscribing:
    # rospy.loginfo('UAV att running!: %d', data.running)
    att_running = data
    # rospy.loginfo('att_running!:  ~~~~~~~~~~~~~~~~~~~~~~ %d ~~~~~~~~~~~~~~~~~~~~', att_running.running)

def local_attitude_setpoint__callback(data):
    global UAV_Att_Setpoint
    # Subscribing:
    # rospy.loginfo('thrust: %f', data.thrust)
    UAV_Att_Setpoint = data

# Publisher:
pub = rospy.Publisher('game_input', Input_Game, queue_size=10)
env_input = Input_Game()
env_input.action = 1    # initial action

current_status = Output_Game()

def status_update(data):
    global current_status
    current_status = data

# terminal functions
def terminal_wo_win_start_in_arrival(posZ, configDict):
    distFromTarget = abs(posZ)
    reward = configDict["rewardFunc"](distFromTarget, configDict)
    
    inTarget = distFromTarget < configDict["zDistSuccess"]
    configDict["inTraget"] |= inTarget
    configDict["steps"] += 1
    configDict["stepsIn"] = configDict["stepsIn"] + 1 if configDict["inTraget"] else configDict["stepsIn"]

    loss = distFromTarget > Z_MAX_DIST
    regularTO = configDict["stepsIn"] >= NUM_STEPS_2_TERMINATE
    timeout = configDict["steps"] >= NON_VALID_NUM_STEPS
    
    termReason = "outOfBounds" if loss else "timeout" if timeout else "normal" if regularTO else ""
    
    reward = -1.0 if loss else reward
    terminal = loss | timeout | regularTO
    
    if terminal:
        configDict["prevDist"] = None
        configDict["inTraget"] = False
    else:
        configDict["prevDist"] = distFromTarget

    return terminal, reward, float(inTarget), termReason

# reward functions
def reward_mid_diff(dist, configDict):
    distSuccess = configDict["zDistSuccess"]
    success = int(dist <= distSuccess)

    r = 0.25 * success
    if not success and "prevDist" in configDict and configDict["prevDist"] != None:
        diff = configDict["prevDist"] - dist
        r += np.clip(diff, -0.05, 0.05)

    return r

# transfer history functions
def no_change_hist(s, a, r, s_, done, prevInterval, currInterval):
    return s, a, r, s_, done

def reset_hist(s, a, r, s_, done, prevInterval, currInterval):
    if prevInterval == currInterval:
        return s, a, r, s_, done
    else:
        return [], [], [], [], []

# reward functions
REWARD_FUNC["RDiffPos"] = reward_mid_diff
REWARD_FUNC["RDiffPosReset"] = reward_mid_diff

# terminal functions
TERMINAL_FUNC["RDiffPos"] = terminal_wo_win_start_in_arrival
TERMINAL_FUNC["RDiffPosReset"] = terminal_wo_win_start_in_arrival

# transfer history function 
TRANSFER_TRANSITIONS["RDiffPos"] = no_change_hist
TRANSFER_TRANSITIONS["RDiffPosReset"] = reset_hist

def get_state(configDict):
    # get UAV status:
    global UAV_Vel, UAV_Pos, UAV_Att_Setpoint

    # 2) judge from current_status: calculate: r, done, failed
    # 3) return current_status, reward, done
    if NORMAL_STATE:
        normalized_pos_z = (UAV_Pos.pose.position.z - TARGET_Z) / Z_NORMAL_VAL      # UAV_Pos.pose.position.z: [10, 30]     -> normalized_pos_z: [][-1, 1]
        normalized_vel_z = UAV_Vel.twist.linear.z / MAX_VELOCITY_Z                 # UAV_Vel.twist.linear.z: [-3, 3]       -> normalized_vel_z: [-1, 1]
        normalized_thrust = (UAV_Att_Setpoint.thrust - THRUST_EQUALIBRIUM) / THRUST_NORMAL     # UAV_Att_Setpoint.thrust: [0.4, 0.78]  -> normalized_thrust: [-1, 1]
    else:
        normalized_pos_z = (UAV_Pos.pose.position.z - TARGET_Z)      # UAV_Pos.pose.position.z: [10, 30]     -> normalized_pos_z: [][-1, 1]
        normalized_vel_z = UAV_Vel.twist.linear.z                 # UAV_Vel.twist.linear.z: [-3, 3]       -> normalized_vel_z: [-1, 1]
        normalized_thrust = (UAV_Att_Setpoint.thrust)      # UAV_Att_Setpoint.thrust: [0.4, 0.78]  -> normalized_thrust: [-1, 1]

    state = np.array((normalized_pos_z, normalized_vel_z, normalized_thrust))
    done, reward, score, terminationReason = configDict["terminalFunc"](UAV_Pos.pose.position.z - TARGET_Z, configDict)
    
    return state, reward, score, done, terminationReason

def actions_2_real_actions(configDict):
    num_action = configDict["numActions"] if "numActions" in configDict else 1
    return np.linspace(0, 1, num_action)

def send_action(action):
    # 1) publish action
    global pub, env_input
    env_input.action = action 
    pub.publish(env_input)

def init_environment():
    rospy.init_node('custom_talker', anonymous=True)
    # 1) get current status:
    # Subscriber:
    rospy.Subscriber('game_status', Output_Game, status_update) 
    # rospy.loginfo('current_status: %f %f %f %f', current_status.vel1, current_status.vel2, current_status.pos1, current_status.pos2)
    # Subscriber:
    rospy.Subscriber('mavros/local_position/pose', PoseStamped, UAV_pos_callback)
    # Subscriber:
    rospy.Subscriber('mavros/local_position/velocity', TwistStamped, UAV_vel_callback)
    # Subscriber:
    rospy.Subscriber('restart_finished_msg', Restart_Finished, restart_finished_callback)
    # Subscriber:
    rospy.Subscriber('att_running_msg', AttControlRunning, att_running_callback)
    # Subscriber:
    rospy.Subscriber('/mavros/setpoint_raw/attitude', AttitudeTarget, local_attitude_setpoint__callback)

def save_config(config, path):
    con = config.copy()
    # delete all objects from config
    del con["terminalFunc"]
    del con["rewardFunc"]
    del con["transferHistFunc"]
    del con["agentClass"]
    open(path, "w+").write(str(con))

def goto_next_run(configDict, first=False):
    if not first:
        path = "./" + configDict["directory"] + "/config"
        configPath = path + ".txt"
        prevConfigPath = path + str(configDict["currRun"]) + ".txt"
        os.system("cp " + configPath + " " + prevConfigPath)
        configDict["currRun"] += 1
        save_config(configDict, configPath)

    if len(configDict["runs"]) > configDict["currRun"]:
        runParams = configDict["runs"][configDict["currRun"]]
    else:
        return False

    configDict["zDistSuccess"] = runParams[0]
    configDict["restartType"] = runParams[1]
    configDict["initZOptions"] = runParams[2]
    configDict["threshold"] = runParams[3]
    configDict["inTraget"] = False
    return True
    

def reset_episode(configDict):
    configDict["steps"] = 0
    configDict["stepsIn"] = 0
    configDict["inBounds"] = False
    configDict["initZ"] = np.random.choice(configDict["initZOptions"]) + TARGET_Z

def read_all_configs(allDirectories, num_state):
    allConfigDict = {}
    startConfig = 0
    configRuns = []
    for i in range(len(allDirectories)):
        directory = allDirectories[i]
        configPath = "./" + directory + "/config.txt"
        configDict = eval(open(configPath, "r+").read())
        configDict["directory"] = directory
        
        if "currRun" not in configDict:
            configDict["currRun"] = 0
        
        goto_next_run(configDict, first=True)
        reset_episode(configDict)

        allConfigDict[directory] = configDict
        runType = configDict["runType"]
        configDict["terminalFunc"] = TERMINAL_FUNC[runType] if runType in TERMINAL_FUNC else TERMINAL_FUNC["default"]
        configDict["rewardFunc"] = REWARD_FUNC[runType] if runType in REWARD_FUNC else REWARD_FUNC["default"]
        configDict["transferHistFunc"] = TRANSFER_TRANSITIONS[runType]


        num_action = configDict["numActions"] if "numActions" in configDict else 1
        agentClass = AGENTS[configDict["agent"]] if "agent" in configDict else AlgoDQN
        agent = agentClass(num_state, num_action, configDict)
        
        configDict["agentClass"] = agent
        configRuns.append((agent.NumEpisodesAll(configDict), i))
    
    # sort according numEpisodes, index
    configRuns.sort()

    return allConfigDict, configRuns[0][1]

def restart_status(restartType):
    if restartType == "realRestart":
        return REAL_RESTART
    elif restartType == "setStartingLoc":
        return SET_STARTING_LOC

# return True if need to switch to next difficulty
def SwitchDifficulty(agent, currConfig):
    resultsKey = currConfig["resultsKey"] if "resultsKey" in currConfig else "reward"
    val = agent.Results(currConfig["numResults"], resultsKey)
    print "curr results =", val, "threshold =", currConfig["threshold"], "num results =", currConfig["numResults"]
    
    return val >= currConfig["threshold"]


def main_train_loop():

    global current_status, pub, env_input

    # get UAV status:
    global UAV_Vel, UAV_Pos, att_running, UAV_Att_Setpoint
    
    init_environment()

    num_state = 3   # state=[UAV_height, UAV_vertical_vel, , UAV_Att_Setpoint.thrust]
    num_actions = 1 # thrust
    
    r = rospy.Rate(ENV_RATE)  # 20Hz
    
    # parse args
    allDirectories = [flags.FLAGS.directoriesPrefix + d for d in flags.FLAGS.directory.split("+")]
    resetModel = flags.FLAGS.resetModel
    resetDtm = flags.FLAGS.resetDtm
    trainEndless = flags.FLAGS.trainEndless
    # num pisodes to change agent
    numEpisodes4Run = flags.FLAGS.numEpisodes
    dtmHistSize = flags.FLAGS.dtmHistSize
    
    # read and init all config
    allConfigDict, configIdx = read_all_configs(allDirectories, num_state)
    currConfig = allConfigDict[allDirectories[configIdx]]
    
    # init dtm
    dtm = DTM(num_state, num_actions, dtmHistSize)
    # init session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # load agents if necessary
    for config in allConfigDict.values():
        config["agentClass"].InitSession(sess, config)
        if not resetModel:
            print "loading model", config["directory"]
            config["agentClass"].load()
    # init dtm
    dtm.InitSession(sess)
    if not resetDtm:
        dtm.load()

    # lut for transition between action to real action
    action2RealAction = actions_2_real_actions(currConfig)
    agent = currConfig["agentClass"]

    print "\n\nrun", "restartType =", currConfig["restartType"], "zDist =", currConfig["zDistSuccess"], "initZOptions", currConfig["initZOptions"],"threshold =", currConfig["threshold"],"\n"

    # init params
    contRun = True
    currAction = None
    actions, states, states_next, rewards, doneAll = [], [], [], [], []
    episodes = 0
    closest = Z_MAX_DIST
    counterStartingLocInit = 0
    status = SET_STARTING_LOC
    scoreAll = 0.0
    sumReward = 0.0
    distAfterIn = 0.0
    distAll = 0.0
    prevTime = time.time()
    while not rospy.is_shutdown() and contRun:
        # start run if not running
        if not (att_running.running):
            # publish random action(0/1) to stop Env-restart(-1) commander!
            env_input.action =  random.randint(0, 1)    # Restart the game!
            # rospy.loginfo('Random action: %f', env_input.action)
            pub.publish(env_input) 
            status = PLAYING 
            reset_episode(currConfig)
            currAction = None
            actions, states, states_next, rewards, doneAll = [], [], [], [], []
        
        else:
            if status == DONE:
                # reset episode and chnge status to play
                reset_episode(currConfig)
                currAction = None
                actions, states, states_next, rewards, doneAll = [], [], [], [], []
                status = PLAYING

            if status == PLAYING:
                # get state
                currState, reward, scoreStep, done, termReason = get_state(currConfig)
                
                # calculate score and reward
                closest = min(abs(currState[0]), closest)
                scoreAll += scoreStep
                sumReward += reward
                distAll += abs(currState[0])
                # if entered to target calculate distAfter
                if currConfig["inTraget"]:
                    distAfterIn += abs(currState[0])
                
                # enter state to history if possible
                if currAction != None:
                    actions.append(currAction)
                    states.append(prevState)
                    states_next.append(currState)
                    rewards.append(reward)
                    doneAll.append(done)

                if not done:
                    # send action
                    currAction = agent.act(currState).squeeze()
                    realAction = currAction if agent.real_action() else action2RealAction[currAction]
                    send_action(realAction)
                    prevState = currState 
                else: 
                    # if non bug in episode insert episode to algorithms
                    nonBug = currConfig["steps"] < NON_VALID_NUM_STEPS or closest < Z_MAX_DIST
                    nonBug = nonBug and currConfig["steps"] > 1
                    if nonBug:
                        # insert episode to algorithms
                        for i in range(len(actions)):
                            dtm.observe(states[i], actions[i], states_next[i], doneAll[i])
                            agent.observe(states[i], actions[i], rewards[i], states_next[i], doneAll[i])
                            agent.replay()   
                        
                        # calculate reward for results monitor and call agents end_episode()
                        reward = float(scoreAll) / NUM_STEPS_2_TERMINATE
                        score = distAfterIn / NUM_STEPS_2_TERMINATE if distAfterIn > 0.0 else distAll / currConfig["steps"]
                        modelSaved, addStr = agent.end_episode(reward, score, currConfig["steps"], sumReward)
                        dtm.end_episode()
                        print "reward =", reward, "score =", float(int(score * 100))/100 , termReason, ", steps =", currConfig["steps"], "initZ =", currConfig["initZ"], "num episode agent =", episodes,"num episode sub agent =", agent.episodes, "realR = ", float(int(sumReward * 100))/100, addStr
                        # switch agent difficulty if necessary
                        if modelSaved and SwitchDifficulty(agent, currConfig):
                            nextRunExist = goto_next_run(currConfig)
                            if nextRunExist:
                                loadModel = not resetModel
                                agent.next_model(currConfig, loadModel)
                                print "\n\ngo to next run", "restartType =", currConfig["restartType"], "zDist =", currConfig["zDistSuccess"], "initZOptions", currConfig["initZOptions"], "threshold =", currConfig["threshold"], "\n"
                    else:
                        print "non valid episode, closest = ", closest
                    
                    # reset params
                    episodes += 1   
                    scoreAll = 0.0
                    sumReward = 0.0
                    distAfterIn = 0.0
                    distAll = 0.0
                    closest = Z_MAX_DIST
                    status = restart_status(currConfig["restartType"])

                    # switch agent if necessary
                    if episodes >= numEpisodes4Run:
                        episodes = 0
                        configIdx += 1
                        if configIdx == len(allConfigDict):
                            if trainEndless:
                                configIdx = 0
                                resetModel = False
                            else:
                                contRun = False
                                break
                                
                        currConfig = allConfigDict[allDirectories[configIdx]]
                        action2RealAction = actions_2_real_actions(currConfig)
                        agent = currConfig["agentClass"]
                        print "\n\nswitch agent dir to", currConfig["directory"], ", run =" , "zDist =", currConfig["zDistSuccess"], "initZOptions", currConfig["initZOptions"], "threshold =", currConfig["threshold"], "\n\n"                                              
   
            elif status == REAL_RESTART:
                send_action(-1)
            
            elif status == SET_STARTING_LOC:
                # send init location for next episode
                send_action(currConfig["initZ"])
                counterStartingLocInit += 1
                if counterStartingLocInit >= 3:
                    status = REAL_RESTART
                    counterStartingLocInit = 0
                
        r.sleep()
                
if __name__ == '__main__':
    flags.FLAGS(sys.argv)
    try:
        main_train_loop()

    except rospy.ROSInterruptException:
        pass
        sys.exit(0)  