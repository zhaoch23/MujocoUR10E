import gym
import numpy as np
import rtde_receive
import rtde_control
import dashboard_client
import time
import numpy as np
import random



class UR10eEnv(gym.Env):
    def __init__(self):
        super(UR10eEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=np.array([-0.05, -0.05, -0.05, -0.05, -0.05, -0.05]), 
                                            high=np.array([0.05, 0.05, 0.05, 0.05, 0.05 , 0.05]), 
                                            dtype=np.float32)  # Joint velocity limits
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # Joint positions

        
        self.FIXED_START = [4.017988204956055, -1.5178674098900338, 2.1686766783343714, -0.8878425520709534,
                            -0.35228139558901006, 0.16946229338645935]
        
        self.HOST = "192.168.0.110"

        self.control = rtde_control.RTDEControlInterface(self.HOST)
        self.receive = rtde_receive.RTDEReceiveInterface(self.HOST)
        self.dashboard = dashboard_client.DashboardClient(self.HOST)
        time.sleep(1)
        self.reconnect()
        self.state = np.zeros(6)  # Initial state
        
    def reconnect(self):

        while not self.control.isConnected():
            print("Control not connected, reconnecting...")
            self.control = rtde_control.RTDEControlInterface(self.HOST)
            time.sleep(5)

        while not self.receive.isConnected():
            print("Receive not connected, reconnecting...")
            self.receive = rtde_receive.RTDEReceiveInterface(self.HOST)
            time.sleep(5)

        while not self.dashboard.isConnected():
            print("Dashboard not connected, reconnecting...")
            self.dashboard = dashboard_client.DashboardClient(self.HOST)
            self.dashboard.connect()
            time.sleep(5)

    def go_to_start(self):
        """Goes back to starting position"""
        start_q = self.FIXED_START
        self.control.moveJ(start_q)


    def reset(self):
        # Reset robot to initial state and return initial observation
        # You need to implement this according to your robot's setup
        # Example: move the robot to a predefined starting position
        self.go_to_start()
        self.state = self.receive.getActualQ()
        return self.state
    
    def step(self, action):
        # Move to pose (linear in joint-space)
        #Parameters
        #pose: target pose
        #speed: joint speed of leading axis [rad/s]
        #acceleration: joint acceleration of leading axis [rad/s^2]
        #asynchronous: a bool specifying if the move command should be asynchronous. If asynchronous is true it is possible to stop a move command using either the stopJ or stopL function. Default is false, this means the function will block until the movement has completed.

        #Actual Cartesian coordinates of the tool: (x,y,z,rx,ry,rz), where rx, ry and rz 
        #is a rotation vector representation of the tool orientation 
        curPose = self.receive.getActualTCPPose()
        self.control.moveJ_IK(curPose+action,speed=0.6, acceleration=0.9, asynchronous=True)
        next_state = self.receive.getActualQ()
        reward = 0  # Placeholder for reward
        done = False  # Placeholder for termination flag
        info = {}  # Placeholder for additional info
        return next_state, reward, done, info

    def close(self):
        self.control.disconnect()
        self.receive.disconnect()
        self.dashboard.disconnect()

if __name__ == "__main__":
    env = UR10eEnv()
    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
    env.stop()
