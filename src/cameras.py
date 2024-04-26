import pyrealsense2 as rs
import numpy as np
import cv2

class BaseCamera(object):

    def wait_for_frame(self):
        '''
        Wait for the next frame to be available.
        
        Returns:
            color_image: The color image frame.
            depth_image: The depth image frame.
        '''
        raise NotImplementedError
    
class RealSenseCamera(BaseCamera):

    def __init__(self, framerate=30) -> None:
        super().__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("Error: Failed to capture RGB frame")
            exit(0)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, framerate)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, framerate)
        self.pipeline.start(self.config)
    
    def wait_for_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame or not depth_frame:
            return None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image
    
class WebCam(BaseCamera):
    def __init__(self, device_id=0, framerate=30) -> None:
        super().__init__()
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FPS, framerate)
        if not self.capture.isOpened():
            print("Error: unable to open the webcam")
            exit(0)
    
    def wait_for_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            print("Error: Failed to capture frame")
            return None, None
        
        return frame, np.zeros_like(frame)
