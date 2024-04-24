import sys
sys.path.append('./src')

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, GestureRecognizerResult, RunningMode
import cv2 

from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib

import numpy as np

from simulation import EnvironmentWrapper
from constants import MODEL_ASSET_PATH
from cameras import BaseCamera


def display_video(frames, framerate=30):
        height, width, _ = frames[0].shape
        dpi = 70
        orig_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
        fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
        matplotlib.use(orig_backend)  # Switch back to the original backend.
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_position([0, 0, 1, 1])
        im = ax.imshow(frames[0])
        def update(frame):
            im.set_data(frame)
            return [im]
        interval = 2000/framerate
        anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                        interval=interval, blit=True, repeat=False)
        
        return anim.to_html5_video()

class HandTrackingSession:


    def __init__(self, 
                 camera: BaseCamera,
                 env_wrapper: EnvironmentWrapper,
                 num_frames=100, 
                 framerate=30,
                 control_scale=0.5,
                 control_timestep=0.2) -> None:
        self.camera = camera
        self.env_wrapper = env_wrapper
        self.num_frames = num_frames
        self.framerate = framerate
        self.control_scale = control_scale
        self.control_timestep = control_timestep
        self.controls = []
        self.camera_frames = []

    def start(self, callback):
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_ASSET_PATH),
            running_mode=RunningMode.IMAGE,
            # result_callback=self.callback
            )
        
        timestamp = 0

        self.env_wrapper.reset()
        self.controls = []
        self.camera_frames = []

        with GestureRecognizer.create_from_options(options) as recognizer:
            try:
                while True:
                    color_frame, depth_frame = self.camera.wait_for_frame()

                    if color_frame is None:
                        continue

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_frame)

                    result = recognizer.recognize(mp_image)

                    if result and result.hand_lanmarks:
                        landmark = result.hand_landmarks[0][0]
                        hand_pos = np.array([landmark.x, landmark.z, landmark.y])
                        
                        # Step the environment
                        ts = self.env_wrapper.step(hand_pos * self.control_scale)
                        joint_pose = self.env_wrapper.get_qpos()
                        
                        # Call the callback
                        callback((color_frame, depth_frame), ts, hand_pos, joint_pose)

                        self.controls.append(hand_pos)
                        self.camera_frames.append((color_frame, depth_frame))

                    if cv2.waitKey(1000 // self.framerate) & 0xFF == ord('q'):
                        break

                    if timestamp == self.num_frames:
                        break

            except KeyboardInterrupt:
                pass