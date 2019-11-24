import os
import subprocess
import time

# run talker itself :
# source ~/src/DRL_node_ROS/devel/setup.bash
# rosrun drl_uav talker.py

def child(cmd):
    print '\nA new child ',  os.getpid(), "cmd =", cmd
    subprocess.call(["bash","-c",cmd])
    os._exit(0)
   
allCmd = ['source ~/src/mavros_ws/devel/setup.bash && roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"', 
            'cd ~/src/Firmware && source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default && make px4_sitl_default gazebo',
            'source ~/src/attitude_controller/devel/setup.bash && rosrun state_machine offb_simulation_test',
            'sleep',
            'rostopic pub -r 20 /mavros/setpoint_position/local geometry_msgs/PoseStamped "{header: {stamp: now, frame_id: \"world\"}, pose: {position: {x: 0, y: 0, z: 2}, orientation: {x: 0.0,y: 0.0,z: 0.0,w: 0.0}}}"',
            'source ~/src/mavros_ws/devel/setup.bash && rosrun mavros mavsys mode -c OFFBOARD',
            'cd ~/src/Firmware && source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default && rosrun mavros mavsafety arm',
            'sleep',
            'killall rostopic',
            ]

for cmd in allCmd:
    if cmd == 'sleep':
        time.sleep(5)
    else:
        newpid = os.fork()
        if newpid == 0:
            child(cmd)
        else:
            pids = (os.getpid(), newpid)
        
        time.sleep(1)

while True:
    continue


