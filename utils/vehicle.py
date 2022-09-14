import numpy as np
from collections import deque


class Vehicle:
    def __init__(self, bbox, pi):
        self.bboxes = deque(maxsize=10)
        self.priority_index = pi
        self.bboxes.append(bbox)
        self.age = 0
    
    def add_bbox(self, bbox):
        self.bboxes.append(bbox)
    
    def set_priority_index(self, pi):
        self.priority_index = pi
    
    def get_priority_index(self):
        return self.priority_index
    
    def get_latest_bbox(self):
        return self.bboxes[-1]
    
    
    