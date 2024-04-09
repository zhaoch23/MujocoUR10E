from math import pi
import os

UR10E_XML_DIR = './assets'
MODEL_ASSET_PATH = "./assets/gesture_recognizer.task"
UR10E_START_POS = [0.0, -pi / 2, pi / 2, 0.0, 0.0, 0.0]
SCENE_XML_PATH = os.path.join(UR10E_XML_DIR, "scene.xml")