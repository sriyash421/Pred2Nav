from queue import Full
from typing import Union
import numpy as np


class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

        self.sim_heading = np.arctan2(self.vy, self.vx)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta]])

    def get_position(self):
        return np.array((self.px, self.py))
    
    def get_velocity(self):
        return self.vx, self.vy

    def get_vpref(self):
        return self.v_pref

    def get_goal(self):
        return self.gx, self.gy

    def get_sim_heading(self):
        return self.sim_heading
    
    def get_heading(self):
        return self.sim_heading

    def set_radius(self, radius):
        self.radius = radius

    def set_goal(self, goal):
        self.gx, self.gy = goal[0], goal[1]

    def set_vpref(self, vpref):
        self.vpref = vpref

    def set_heading(self, heading):
        self.heading = heading

    def set_sim_heading(self, sim_heading):
        self.sim_heading = sim_heading


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])

    def get_position(self):
        return self.px, self.py

    def set_position(self, pos):
        self.px, self.py = pos[0], pos[1]

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, vel):
        self.vx, self.vy = vel[0], vel[1]

    def get_radius(self):
        return self.radius


class JointState(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.self_state = self_state
        self.human_states = human_states


class ObservableState_noV(object):
    def __init__(self, px, py, radius):
        self.px = px
        self.py = py
        self.radius = radius

        self.position = (self.px, self.py)

    def __add__(self, other):
        return other + (self.px, self.py, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.radius]])


class BallbotState(FullState):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta, lean_theta, lean_theta_dot):
        super().__init__(px, py, vx, vy, radius, gx, gy, v_pref, theta)
        self.lean_theta = lean_theta
        self.lean_theta_dot = lean_theta_dot
    
    def get_theta(self):
        return self.lean_theta
    
    def get_theta_dot(self):
        return self.lean_theta_dot

class JointState_noV(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, Union(FullState, BallbotState))
        for human_state in human_states:
            assert isinstance(human_state, ObservableState_noV)

        self.self_state = self_state
        self.human_states = human_states