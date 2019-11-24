import math
import numpy as np
import pandas as pd
import pickle
import os.path
import time
import datetime
import sys
import threading
import os
import operator

import matplotlib
from matplotlib import pyplot as plt

from multiprocessing import Process, Lock, Value, Array, Manager

DefaultColorList = []
def AvgResults(path, name, idxDir=None, key4Results=None):
    # create results read only class according to args
    if idxDir == None:
        results = ReadOnlyResults(path + '/' + name)
    else:
        results = ReadOnlyResults(path + '_' + str(idxDir) + '/' + name)

    # return avg reward
    return results.AvgReward(key4Results)

def ChangeName2NextResultFile(path, name, idxDir, idx2CurrentFile):
    # 
    srcFName = path + '_' + str(idxDir) + '/' + name + ".gz"
    newFName = path + '_' + str(idxDir) + '/' + name + "_" + str(idx2CurrentFile) +".gz"
    
    # if results exist reanme file name to newFName
    if os.path.isfile(srcFName):
        if os.path.isfile(newFName):
            os.remove(newFName)
        os.rename(srcFName, newFName)
        return True

    return False

def PlotMeanWithInterval(x, y, interval, axes=plt, color=None):
    if color != None:
        # plot the shaded range of the confidence intervals
        axes.fill_between(x, y + interval, y - interval, color=color, alpha=.5)
        # plot the mean on top
        axes.plot(x, y, color=color)
    else:
        # plot the shaded range of the confidence intervals
        axes.fill_between(x, y + interval, y - interval, alpha=.5)
        # plot the mean on top
        axes.plot(x, y)

def PlotResults(resultsDirNames=[], keys2Plot=["reward"], grouping=10, isFNamePrefix=True, plotInDiffFig=False, orderGroups=[], barPlot=False, maxEpisode=None):
    print resultsDirNames
    # plot results
    numPlots = len(keys2Plot) * (1 + barPlot)
    plot = PlotMngr(resultsDirNames, numPlots, maxTrialsToPlot=maxEpisode)
    if plot.initFailed:
        return

    if plotInDiffFig:
        for fname in plot.resultFileNames:
            idxPlot = 1
            for key in keys2Plot:
                idxPlot = plot.Plot(grouping, key, idxPlot, resultFileNames=[fname])
            
            plot.Save(fname.replace("/","_").replace(".gz","").replace(".h5","").replace(".",""))
            plot.CreateNewFig()
            plt.show()

    else:
        idxPlot = 1
        for key in keys2Plot:
            if len(orderGroups) > 0:
                idxPlot = plot.PlotGroups(grouping, key, idxPlot, orderGroups)
            else:
                idxPlot = plot.Plot(grouping, key, idxPlot)
            if len(orderGroups) > 0 and barPlot:
                idxPlot = plot.BarPlotGroups(grouping, key, idxPlot, orderGroups)

        plot.Save()
        plt.show()


class PlotMngr:
    def __init__(self, resultDirList, numPlots, resultsFName="talkerResults", maxTrialsToPlot=None):
        # list of all necessary ReadOnlyResults class (in case of multiple decision maker the list is 2 dimensions)
        self.initFailed = False
        self.plotLast2Max = False
        self.idxKeys = {"reward": 1, "score": 2, "steps" : 3, "realReward" : 4}
        self.resultFileList = {}
        self.resultFileNames = {}
        self.maxTrialsToPlot = maxTrialsToPlot

        for dirName in resultDirList:
            self.resultFileNames[dirName] = []
            files = os.listdir(dirName)
            for f in files:
                if f.find("Results") >= 0:
                    self.resultFileNames[dirName].append("./" + dirName + "/" + f)

        
        allFiles = []
        for dirName in resultDirList:
            self.resultFileList[dirName] = {}
            for fname in self.resultFileNames[dirName]:
                allFiles.append(fname)
                self.resultFileList[dirName][fname] = ResultFile(fname.replace(".gz", ""), 10) #ReadOnlyResults(fname)
                self.resultFileList[dirName][fname].Load()
                print fname, "size =", self.resultFileList[dirName][fname].table.shape, "results =", self.resultFileList[dirName][fname].Results(50)



        self.legend = {}
        
        if len(allFiles) > 1:
            self.title = allFiles[0].replace(".gz", "")
            for name in allFiles:
                while not name.startswith(self.title):
                    self.title = self.title[:-1]
            
            if self.title.rfind("_") > 0:
                self.title = self.title[:self.title.rfind("_")]
            
            for name in allFiles:
                nameClean = name[len(self.title):]
                nameClean = nameClean.replace(".gz", "").replace(" ","").replace("_","").replace("/","")
                nameClean = nameClean.replace(resultsFName,"")
                self.legend[name] = nameClean

            self.title = self.title.replace("./", "")
        elif len(allFiles) == 1:
            self.title = allFiles[0]
        else:
            print "\n\n files not exist"
            self.initFailed = True
            return 
        print "\n\n"
        
        self.title = self.title.replace("/", "_")
        self.title = self.title.replace("talker", "")
        
        dirList = resultDirList
        self.plotFName = "./results/results"
        for dn in resultDirList:
            self.plotFName += "_" + dn.replace("/","")

        self.numRows = int(math.ceil(float(numPlots) / 2))
        self.numCols = 2 if numPlots > 1 else 1

        self.CreateNewFig()
        

    def CreateNewFig(self):
        self.fig = plt.figure(figsize=(19.0, 11.0))
        self.fig.suptitle("results for RL agent: " + self.title, fontsize=20)

    def Plot(self, grouping, key, idxPlot, resultFileNames=[]):
        plt.subplot(self.numRows,self.numCols,idxPlot)
        idxPlot += 1
        legend = []
        for dirName, rFNameList in self.resultFileNames.items():
            for rFName in rFNameList:
                if len(resultFileNames) > 0 and rFName not in resultFileNames:
                    continue

                results, t = self.ResultsFromTable(self.resultFileList[dirName][rFName].table, grouping, key) 
                plt.plot(t, results)
                legend.append(self.legend[rFName])

        plt.ylabel('avg ' + key + ' for ' + str(grouping) + ' trials')
        plt.xlabel('#trials')
        plt.title(self.title + ': ' + key)
        plt.grid(True)
        plt.legend(legend, loc='best')

        return idxPlot

    def BarPlotGroups(self, grouping, key, idxPlot, orderGroups):
        plt.subplot(self.numRows,self.numCols,idxPlot)
        idxPlot += 1
        legend = []

        resultsAvg = []
        resultsStd = []

        for dirName, groupFileNames in self.resultFileNames.items():
            for singleLoc in orderGroups:
                searchLoc = [fname for fname in groupFileNames if fname.find(singleLoc) >= 0]
                if len(searchLoc) > 0:
                    rFName = searchLoc[0]
                    results, t = self.ResultsFromTable(self.resultFileList[dirName][rFName].table, grouping, key) 
                    if len(t) > 0:
                        resultsAvg.append(np.average(results))
                        resultsStd.append(np.std(results))
                        legend.append(self.legend[rFName])

        x = np.arange(len(resultsAvg))
        for i in range(len(resultsAvg)):
            y = np.zeros(len(resultsAvg))
            y[i] = resultsAvg[i]
            err = np.zeros(len(resultsAvg))
            err[i] = resultsStd[i]
            plt.bar(x,y, yerr=err, align='center', alpha=0.75, ecolor='r', capsize=2)
        
        plt.title("bar results for " + key + " key")
        plt.ylabel(key)
        plt.legend(legend)

        return idxPlot

    def PlotGroups(self, grouping, key, idxPlot, orderGroups):
        plt.subplot(self.numRows,self.numCols,idxPlot)
        
        allResults = dict((group,{}) for group in orderGroups)
        t2Add = dict((group,0) for group in orderGroups)
        for group in orderGroups:
            for dirName, groupFileNames in self.resultFileNames.items():                
                searchLoc = [fname for fname in groupFileNames if fname.find(group) >= 0]
                
                if len(searchLoc) > 1:
                    searchLoc = sorted(searchLoc, key=lambda x: x.find(group))

                if len(searchLoc) > 0:
                    rFName = searchLoc[0]
                    
                    results, t = self.ResultsFromTable(self.resultFileList[dirName][rFName].table, grouping, key) 
                    if len(t) > 0:
                        if key == "score":
                            results /= 2
                        
                        allResults[group][dirName] = {"results" : results, "t": t}
                        t2Add[group] = max(t2Add[group], t[-1])
                        

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] + ['r', 'm', 'c', 'b']
        dirIdx = 1
        legend = []

        
        
        for dirName in self.resultFileNames.keys():
            if dirName != "PID" and self.plotLast2Max:
                lastGroup = max(t2Add.iteritems(), key=operator.itemgetter(1))[0]
            else:
                lastGroup = "none"
            lastT = 0
            idx = 0
            
            for group in orderGroups:
                if dirName in allResults[group]:
                    legend.append(dirName + " " + group)
                    t = allResults[group][dirName]["t"] + lastT
                    
                    if group == lastGroup:
                        print("lastGroup =", lastGroup)
                        maxIdx = np.argmax(allResults[group][dirName]["results"])
                        plt.plot(t[:maxIdx], allResults[group][dirName]["results"][:maxIdx], colors[idx], alpha=1.0/dirIdx)
                    else:
                        plt.plot(t, allResults[group][dirName]["results"], colors[idx], alpha=1.0/dirIdx)
                 
                    lastT += t2Add[group]
                    idx = idx + 1

            dirIdx += 1
            
        if key == "score":
            plt.ylabel('precentage in target location')
        else:
            plt.ylabel('avg ' + key + ' for ' + str(grouping) + ' trials')
        plt.xlabel('#trials')
        plt.title(self.title + ': ' + key)
        plt.grid(True)
        if key == "steps":
            plt.legend(legend, loc='best')

        idxPlot += 1
        return idxPlot


    def Save(self, resultFName=None):
        # save figure
        if resultFName == None:
            fname = self.plotFName + ".png"
        else:
            fname = "./results/results_" + resultFName + ".png"
        
        
        self.fig.savefig(fname)
        print "results graph saved in:", fname


    def ResultsFromTable(self, table, grouping, key, maxTrials2Plot=None, groupSizeIdx=0):
        if key.find("/") >= 0: 
            allKeys = key.split("/")
            r1, t = self.ResultsFromTable(table, grouping, allKeys[0], maxTrials2Plot, groupSizeIdx) 
            r2, t = self.ResultsFromTable(table, grouping, allKeys[1], maxTrials2Plot, groupSizeIdx)
            return r1 / r2, t

        dataIdx = self.idxKeys[key]

        names = list(table.index)
        tableSize = len(names) -1
        
        resultsRaw = np.zeros((2, tableSize), dtype  = float)
        sumRuns = 0
        realSize = 0

        # min grouping in table in case its changed during run
        minSubGrouping = grouping
        if maxTrials2Plot == None:
            maxTrials2Plot = self.maxTrialsToPlot

        for name in names[:]:
            if name.isdigit():
                idx = int(name)
                subGroupSize = table.ix[name, groupSizeIdx]
                minSubGrouping = min(subGroupSize, minSubGrouping)
                resultsRaw[0, idx] = subGroupSize
                resultsRaw[1, idx] = table.ix[name, dataIdx]

                sumRuns += subGroupSize
                realSize += 1
                if maxTrials2Plot != None and maxTrials2Plot < sumRuns:
                    break
  
        # create results for equal grouping
        results = np.zeros( int(sumRuns / minSubGrouping) , dtype  = float)

        offset = 0
        for idx in range(realSize):
            
            subGroupSize = resultsRaw[0, idx]
            for i in range(int(subGroupSize / minSubGrouping)):
                results[offset] = resultsRaw[1, idx]
                offset += 1
        
        # transfer results to requested grouping
        
        groupSizes = int(math.ceil(grouping / minSubGrouping))
        # for calculation of average of group
        idxArray = np.arange(groupSizes)
        
        groupResults = []
        timeLine = []
        t = 0
        startIdx = groupSizes - 1
        for i in range(startIdx, len(results)):
            groupResults.append(np.average(results[idxArray]))
            timeLine.append(t)
            idxArray += 1
            t += minSubGrouping

        return np.array(groupResults), np.array(timeLine)    

class ReadOnlyResults():
    def __init__(self, tableName):
        
        self.countIdx = 0
        self.rewardIdx = 1
        self.scoreIdx = 2
        self.stepsIdx = 3
        self.realRewardIdx = 4
        self.rewardCol = list(range(5))

        self.tableName = tableName
        self.table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        if tableName.find(".gz") < 0:
            tableName += ".gz"
        if os.path.isfile(tableName) and os.path.getsize(tableName) > 0:
            self.table = pd.read_pickle(tableName, compression='gzip')
        else:
            print "Error table ", tableName,"not found"
            
    
    def AvgReward(self, key = None):
        names = list(self.table.index)
        if len(names) == 0:
            return None
        
        # calulate sum reward and count
        sumVal = 0.0
        count = 0
        for i in range(len(names)):
            k = str(i)
            if k in self.table.index:
                v = self.table.ix[k, self.rewardIdx]
                c = self.table.ix[k, self.countIdx]
                sumVal += v * c
                count += c
        
        return sumVal / count if count > 0 else None           


class ResultFile:
    def __init__(self, tableName, numToWrite=100, agentName = ''):
        self.saveFileName = tableName
                
        self.numToWrite = numToWrite
        self.agentName = agentName

        self.countCompleteKey = 'countComplete'
        self.countIdx = 0
        self.rewardIdx = 1
        self.scoreIdx = 2
        self.stepsIdx = 3
        self.realRewardIdx = 4
        self.rewardCol = list(range(5))

        self.idxKeys = {"count" : self.countIdx, "reward" : self.rewardIdx, 
                        "steps" : self.stepsIdx, "score" : self.scoreIdx, "realReward" : self.realRewardIdx}

        self.sumReward = 0
        self.sumScore = 0
        self.sumSteps = 0
        self.sumRealReward = 0
        self.numRuns = 0


        self.InitTable()

    def InitTable(self):
        # init table and create complete count key
        self.table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        self.check_state_exist(self.countCompleteKey)
        self.countComplete = int(self.table.ix[self.countCompleteKey, 0])

    def Load(self):
        # if file exist reaf table and read from table complete count key
        if os.path.isfile(self.saveFileName + '.gz') and os.path.getsize(self.saveFileName + '.gz') > 0:
            self.table = pd.read_pickle(self.saveFileName + '.gz', compression='gzip')
            self.countComplete = int(self.table.ix[self.countCompleteKey, 0])
            return True

        return False

    def Save(self, saveName=""):
        saveName = self.saveFileName if saveName == "" else saveName
        print "save results file in", saveName
        self.table.to_pickle(saveName + '.gz', 'gzip') 
    
    def NumRuns(self):
        # if num to write changes during run this function return wrong results
        return self.countComplete * self.numToWrite + self.numRuns

    def check_state_exist(self, state):
        if state not in self.table.index:
            # append new state filled with 0
            self.table = self.table.append(pd.Series([0] * len(self.rewardCol), index=self.table.columns, name=state))
            return True
        
        return False

    def insertEndRun2Table(self):
        # calculate results
        avgReward = self.sumReward / self.numRuns
        avgScore = self.sumScore / self.numRuns
        avgSteps = self.sumSteps / self.numRuns
        avgRealReward = self.sumRealReward / self.numRuns
        
        # find key and create state
        countKey = str(self.countComplete)
        self.check_state_exist(countKey)

        # insert values to table
        self.table.ix[countKey, self.countIdx] = self.numRuns
        self.table.ix[countKey, self.rewardIdx] = avgReward
        self.table.ix[countKey, self.scoreIdx] = avgScore
        self.table.ix[countKey, self.stepsIdx] = avgSteps
        self.table.ix[countKey, self.realRewardIdx] = avgRealReward

        # update count complete key
        self.countComplete += 1
        self.table.ix[self.countCompleteKey, 0] = self.countComplete

        # reset current results
        self.sumReward = 0
        self.sumRealReward = 0
        self.sumScore = 0
        self.numRuns = 0
        self.sumSteps = 0

        #print "\t\t", threading.current_thread().getName(), ":", self.agentName, "->avg results for", self.numToWrite, "trials: reward =", avgReward, "score =", avgScore, "numRun =", self.NumRuns()

    def end_run(self, r, score, steps, realReward):
        # insert results
        self.sumSteps += steps
        self.sumReward += r
        self.sumRealReward += realReward
        self.sumScore += score
        self.numRuns += 1

        saved = False
        # save curr results in table if necessary
        if self.numRuns == self.numToWrite:
            self.insertEndRun2Table()        
            self.Save()
            saved = True
        
        
        return saved

    def AddSlot(self, slotName):
        # add slot for switching location
        if self.check_state_exist(slotName):
            self.table.ix[slotName, 0] = self.countComplete

    def Reset(self):
        self.InitTable()

        # reset result that not saved to table
        self.sumReward = 0
        self.sumScore = 0
        self.sumSteps = 0
        self.numRuns = 0

    def Results(self, size, key="reward"):
        if size > self.countComplete * self.numToWrite:
            return -1
        else:
            idxKey = self.idxKeys[key]
            num = int(size / self.numToWrite)
            sumR = 0.0
            for i in range(num):
                idx = self.countComplete - i - 1
                sumR += self.table.ix[str(idx), idxKey]
            return sumR / num

    def LastResults(self, size, key="reward"):
        if size > self.countComplete * self.numToWrite:
            return []
        else:
            idxKey = self.idxKeys[key]
            num = int(size / self.numToWrite)
            allIdx = [str(i) for i in range(self.countComplete - num,self.countComplete)]
            results = self.table.ix[allIdx, idxKey]

            return results

if __name__ == "__main__":
    from absl import flags
    flags.DEFINE_string("grouping", "50", "")

    flags.DEFINE_string("directory", "", "")
    flags.DEFINE_string("directoriesPrefix", "", "model directory")
    flags.DEFINE_string("directoriesEnd", "", "model directory")
    flags.DEFINE_string("keys2Plot", "reward+score+steps+realReward", "")
    flags.DEFINE_bool("isFNamePrefix", True, "")
    flags.DEFINE_bool("barPlot", False, "")
    flags.DEFINE_string("plotInDiffFig", "False", "")
    flags.DEFINE_string("maxEpisode", "none", "")
    flags.DEFINE_bool("fromCurrRun", False, "")
    flags.DEFINE_bool("allInOne", False, "")
    
    flags.FLAGS(sys.argv)

    isFNamePrefix = flags.FLAGS.isFNamePrefix
    plotInDiffFig = eval(flags.FLAGS.plotInDiffFig)
    grouping = int(flags.FLAGS.grouping)
    
    directoriesEnd = flags.FLAGS.directoriesEnd.split("+")
    directories = []
    
    for endName in directoriesEnd:
        directories += [flags.FLAGS.directoriesPrefix + d + endName for d in flags.FLAGS.directory.split("+")]

    
    keys2Plot = flags.FLAGS.keys2Plot.split("+")

    maxEpisode = None if flags.FLAGS.maxEpisode == "none" else int(flags.FLAGS.maxEpisode)

    orderGroups = []
    if not plotInDiffFig:
        allConfig = []
        for directory in directories:
            configPath = "./" + directory + "/config.txt"
            allConfig.append(eval(open(configPath, "r+").read()))

        if len (allConfig) > 1:
            for config in allConfig:
                for run in config["runs"]:
                    runName = str(run[0]) + "_" + str(run[1]) + "_" + str(run[2])
                    orderGroups.append(runName.replace(" ",""))
            
            orderGroups = [orderGroups[i] for i in range(len(orderGroups)) if orderGroups[i] not in orderGroups[:i]]

        else:
            config = allConfig[0]
            startRun = config["currRun"] if flags.FLAGS.fromCurrRun else 0
            for i in range(startRun, len(config["runs"])):
                run = config["runs"][i]
                runName = str(run[0]) + "_" + str(run[1]) + "_" + str(run[2])
                orderGroups.append(runName.replace(" ",""))
    
    if flags.FLAGS.allInOne:
        PlotResults(directories, keys2Plot=keys2Plot, grouping=grouping, isFNamePrefix=isFNamePrefix, plotInDiffFig=plotInDiffFig, 
                    orderGroups=orderGroups, barPlot=flags.FLAGS.barPlot, maxEpisode=maxEpisode)
    else:
        for d in directories:
            PlotResults([d], keys2Plot=keys2Plot, grouping=grouping, isFNamePrefix=isFNamePrefix, plotInDiffFig=plotInDiffFig, 
                    orderGroups=orderGroups, barPlot=flags.FLAGS.barPlot, maxEpisode=maxEpisode)