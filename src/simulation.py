import sys
sys.path.append('./src')

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import *


class SimTask(base.Task):

    def __init__(self, init_pos, random=None):
        super(SimTask, self).__init__(random=random)

        self.init_pos = init_pos

        self.init_mocap_pos = None
        self.init_mocap_quat = None
        self.simulation_frames = None

    def initialize_episode(self, physics):
        with physics.reset_context():
            physics.named.data.qpos[:6] = self.init_pos
            np.copyto(physics.data.ctrl, self.init_pos)
        
        np.copyto(physics.data.mocap_pos[0], physics.named.data.xpos['wrist_3_link'])
        np.copyto(physics.data.mocap_quat[0], physics.named.data.xquat['wrist_3_link'])

        super().initialize_episode(physics)

    def before_step(self, action, physics):
        np.copyto(physics.data.mocap_pos[0], action[0:3])
        np.copyto(physics.data.mocap_quat[0], action[3:7])
    
    def get_observation(self, physics):
        return {
            'images': {
                'top': physics.render(height=240, width=320, camera_id=0),
                'birdview': physics.render(height=240, width=320, camera_id=1),
                'front': physics.render(height=240, width=320, camera_id=2)
            }, 'mocap': {
                'pos': physics.data.mocap_pos[0],
                'quat': physics.data.mocap_quat[0]
            }
        }
    
    def get_reward(self, physics):
        return 0
    
    def get_joint_poses(self, physics):
        # Get the joint positions from the Mujoco physics simulation
        joint_poses = physics.named.data.qpos[6:]
        return joint_poses
    
    @staticmethod
    def get_env(init_pos, control_timestamp_sec=0.02):
        physics = mujoco.Physics.from_xml_path(SCENE_XML_PATH)

        task = SimTask(init_pos)
        env = control.Environment(
            physics, task, time_limit=10., control_timestep=control_timestamp_sec,
            n_sub_steps=None, flat_observation=None)
        return env


class EnvironmentWrapper:

    def __init__(self, init_pos, control_timestamp_sec=0.02):
        self.env = SimTask.get_env(init_pos, control_timestamp_sec)

        self.init_mocap_pos = None
        self.init_mocap_quat = None

    def reset(self):
        ts = self.env.reset()
        self.init_mocap_pos = ts.observation['mocap']['pos'].copy()
        self.init_mocap_quat = ts.observation['mocap']['quat'].copy()

        return ts

    def step(self, xyz):
        return self.env.step(
            np.concatenate([xyz + self.init_mocap_pos, self.init_mocap_quat]))

    def get_qpos(self):
        return self.env.physics.data.qpos.copy()

    def close(self):
        self.env.close()