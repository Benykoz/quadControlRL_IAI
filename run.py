import sys
import os

# source ~/src/DRL_node_ROS/devel/setup.bash
if  __name__ == "__main__":
    runCmd = "rosrun drl_uav talker.py"
    for arg in sys.argv[1:]:
        runCmd += " " + arg

    idx = 0
    cpCmd = "cp -r ./ ~/backups/b_"
    crashFile = "./crash.txt"
    crash = False
    while True:
        stat = os.system(runCmd)
        if stat != 0:
            print "exit status =", stat
            crash = eval(open(crashFile, "r+").read())
        os.system(cpCmd + str(idx))
        idx += 1
        if crash:
            exit()
    