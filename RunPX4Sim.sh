#!/usr/bin/env bash

me=$(realpath $0)

allTerminals=(
# terminal 1
'source ~/src/mavros_ws/devel/setup.bash && roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"'
# terminal 2
'cd ~/src/Firmware && source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default && make gazebo'
# terminal 3
'source ~/src/attitude_controller/devel/setup.bash && rosrun state_machine offb_simulation_test'
# terminal 4
'rostopic pub -r 20 /mavros/setpoint_position/local geometry_msgs/PoseStamped "{header: {stamp: now, frame_id: \"world\"}, pose: {position: {x: 0, y: 0, z: 2}, orientation: {x: 0.0,y: 0.0,z: 0.0,w: 0.0}}}"'
# terminal 5
'source ~/src/mavros_ws/devel/setup.bash && rosrun mavros mavsys mode -c OFFBOARD'
# terminal 6
'cd ~/src/Firmware && source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default && rosrun mavros mavsafety arm'
# terminal 7
'killall rostopic'
)
# in code:
#  source ~/src/DRL_node_ROS/devel/setup.bash && rosrun drl_uav talker.py

len=${#allTerminals[@]}

if [ $# -eq 0 ];then
    num=$len
else
    num=$1
fi

n=$((len-num))
commands=${allTerminals[$n]}
num=$((num-1))

xdotool key ctrl+shift+t

if [ $num -gt -1 ]; then
	xdotool type --delay 1 --clearmodifiers "$me $num; ${commands}"; 
	#sleep 20; 
	
#xdotool key Return;
fi
 
