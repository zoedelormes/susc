import math

y0 = 55
A = 25
t0 = 200
T = 365

def model_value(t): 
    model= y0 + A* math.cos(2*math.pi)*(t-t0 /T)
    return model

import pandas as pd
import numpy as np

t = np.array([])

for i in range(1,365):
    t = np.append(t,i)

model_value(t)
