import os
import numpy as np
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

class RealHuman(Policy):
    def __init__(self, config):
        super().__init__(config)
        self.ep = 0
        self.t = 0

    def set_human_id(self, id):
        self.hid = id
    
    def reset(self, id, ep):
        temp = np.load(os.path.join('/home/sriyash421/PRL22/CrowdNav_DSRNN/crowd_nav/policy/cv_trajs2', f'{ep}_trajectory.npy'))
        print(temp.shape)
        self._steps_array = temp[:, id]
        self.t = 0

    def predict(self, state):
        """
        Produce action for agent with circular specification of social force model.
        """
        vx, vy = self._steps_array[self.t, 2:4]
        self.t+=1
        act_norm = np.linalg.norm([vx, vy])
        if act_norm > state.self_state.v_pref:
            return ActionXY(vx / act_norm * state.self_state.v_pref, vy / act_norm * state.self_state.v_pref)
        else:
            return ActionXY(vx, vy)
