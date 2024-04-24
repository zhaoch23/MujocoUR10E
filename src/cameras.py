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

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError # Implement this
    
    def wait_for_frame(self):
        raise NotImplementedError