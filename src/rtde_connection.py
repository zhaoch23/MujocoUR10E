import os
import psutil
import sys
import math

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive


class RTDEConnection(object):
    '''
    Wrapper class for RTDEControl and RTDEReceive
    '''

    def __init__(self, 
                 robot_ip,
                vel = 0.5,
                acc = 0.5,
                rtde_frequency = 500.0,
                flags = RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT,
                ur_cap_port = 50002,
                lookahead_time = 0.1,
                gain = 600,
                 ) -> None:
        self.robot_ip = robot_ip
        self.vel = vel
        self.acc = acc
        self.rtde_frequency = rtde_frequency
        self.dt = 1.0/rtde_frequency
        self.flags = flags
        self.ur_cap_port = ur_cap_port
        self.lookahead_time = lookahead_time
        self.gain = gain

        self._t_start = None
        
    def connect(self):
        # ur_rtde realtime priorities
        rt_receive_priority = 90
        rt_control_priority = 85
        self.rtde_r = RTDEReceive(self.robot_ip, self.rtde_frequency, [], True, False, rt_receive_priority)
        self.rtde_c = RTDEControl(self.robot_ip, self.rtde_frequency, self.flags, self.ur_cap_port, rt_control_priority)

        # Set application real-time priority
        os_used = sys.platform
        process = psutil.Process(os.getpid())
        if os_used == "win32":  # Windows (either 32-bit or 64-bit)
            process.nice(psutil.REALTIME_PRIORITY_CLASS)
        elif os_used == "linux":  # linux
            rt_app_priority = 80
            param = os.sched_param(rt_app_priority)
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, param)
            except OSError:
                print("Failed to set real-time process scheduler to %u, priority %u" % (os.SCHED_FIFO, rt_app_priority))
            else:
                print("Process real-time priority set to: %u" % rt_app_priority)

    def schedule(self, func):
        '''
        Schedule a function to be executed in the next cycle
        '''
        if self._t_start is None:
            self._t_start = self.rtde_c.initPeriod()
            func()
        else:
            self.rtde_c.waitPeriod(self._t_start)
            self._t_start = self.rtde_c.initPeriod()
            func()

    def is_connected(self):
        return self.rtde_c.isConnected() and self.rtde_r.isConnected()

    def stop(self):
        if self.is_connected():
            self.rtde_c.servoStop()
            self.rtde_c.stopScript()
        else:
            raise Exception("Not connected to the robot")
    
    def get_actual_tcp_pose(self):
        if not self.is_connected():
            raise Exception("Not connected to the robot")
        return self.rtde_r.getActualTCPPose()
    
    def moveL(self, pose):
        if not self.is_connected():
            raise Exception("Not connected to the robot")
        self.schedule(lambda: self.rtde_c.moveL(pose, self.vel, self.acc))
    
    def servoL(self, pose):
        if not self.is_connected():
            raise Exception("Not connected to the robot")
        self.schedule(lambda: self.rtde_c.servoL(pose, self.vel, self.acc, self.dt, self.lookahead_time, self.gain))

def getCircleTarget(pose, timestep, radius=0.075, freq=1.0):
    circ_target = pose[:]
    circ_target[0] = pose[0] + radius * math.cos((2 * math.pi * freq * timestep))
    circ_target[1] = pose[1] + radius * math.sin((2 * math.pi * freq * timestep))
    return circ_target

if __name__ == "__main__":
    robot_ip = "192.168.17.128" # Change this to the robot's IP
    
    rtde_connection = RTDEConnection(robot_ip)
    rtde_connection.connect()

    actual_tcp_pose = rtde_connection.get_actual_tcp_pose()
    init_pose = getCircleTarget(actual_tcp_pose, 0)
    rtde_connection.moveL(init_pose)

    time_counter = 0.0
    try:
        while True:
            servo_target = getCircleTarget(actual_tcp_pose, time_counter)
            print(f"Target: {servo_target}")
            rtde_connection.servoL(servo_target)
            time_counter += rtde_connection.dt
    except KeyboardInterrupt:
        print("Control Interrupted!")
        rtde_connection.stop()