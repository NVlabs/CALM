# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

from isaacgym.torch_utils import *

from env.tasks.humanoid_strike import HumanoidStrike
from utils import torch_utils
from enum import Enum

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2


class MovementType(Enum):
    RUN = 0
    WALK = 1
    CROUCH = 2


class AttackType(Enum):
    KICK = 3
    SWORD_SWIPE = 4
    SHIELD_SWIPE = 5
    SHIELD_CHARGE = 6


class HumanoidStrikeFSM(HumanoidStrike):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._should_strike = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._stay_idle = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._tar_height = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)

        self.movement_type = MovementType[cfg["env"]["movementType"].upper()]
        self.attack_type = AttackType[cfg["env"]["attackType"].upper()]

        self.motions_dict = {
            AttackType.KICK: {
                MovementType.RUN: 50,
                MovementType.WALK: 11,
                MovementType.CROUCH: 9
            },
            AttackType.SWORD_SWIPE: {
                MovementType.RUN: 22,
                MovementType.WALK: 8,
                MovementType.CROUCH: 5.5
            },
            AttackType.SHIELD_SWIPE: {
                MovementType.RUN: 22,
                MovementType.WALK: 8,
                MovementType.CROUCH: 5.5
            },
            AttackType.SHIELD_CHARGE: {
                MovementType.RUN: 15,
                MovementType.WALK: 7,
                MovementType.CROUCH: 7
            },
        }

        self._idle_index = 7
        self._strike_index = self.attack_type.value

        self._enable_early_termination = False
        self._near_prob = 0

        self._tar_dist_min = 10.0
        self._tar_dist_max = 10.0

        self._total_episodes = None
        self._total_successes = 0

        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 3

        return obs_size

    def _reset_target(self, env_ids):
        n = len(env_ids)

        if self._total_episodes is not None:
            if n > 0:
                tar_pos = self._target_states[..., 0:3][env_ids]
                tar_rot = self._target_states[..., 3:7][env_ids]

                up = torch.zeros_like(tar_pos)
                up[..., -1] = 1
                tar_up = quat_rotate(tar_rot, up)
                tar_rot_err = torch.sum(up * tar_up, dim=-1)

                succ = tar_rot_err < 0.2

                self._total_episodes += n
                self._total_successes += torch.sum(succ.long())

                print(f'Current success rate: {self._total_successes * 100. / self._total_episodes}%, out of {self._total_episodes} episodes')
        else:
            self._total_episodes = 0

        ########

        init_near = torch.rand([n], dtype=self._target_states.dtype,
                               device=self._target_states.device) < self._near_prob
        dist_max = self._tar_dist_max * torch.ones([n], dtype=self._target_states.dtype,
                                                   device=self._target_states.device)
        dist_max[init_near] = self._near_dist
        rand_dist = (dist_max - self._tar_dist_min) * torch.rand([n], dtype=self._target_states.dtype,
                                                                 device=self._target_states.device) + self._tar_dist_min

        rand_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        self._target_states[env_ids, 0] = rand_dist * torch.cos(rand_theta) + self._humanoid_root_states[env_ids, 0]
        self._target_states[env_ids, 1] = rand_dist * torch.sin(rand_theta) + self._humanoid_root_states[env_ids, 1]
        self._target_states[env_ids, 2] = 0.9

        rand_rot_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)

        self._target_states[env_ids, 3:7] = rand_rot
        self._target_states[env_ids, 7:10] = 0.0
        self._target_states[env_ids, 10:13] = 0.0
        self._should_strike[env_ids] = 0
        self._stay_idle[env_ids] = 0

        self._tar_height[env_ids] = self.movement_type.value
        return

    def _compute_task_obs(self, env_ids=None):
        root_pos = self._humanoid_root_states[..., 0:3]
        tar_pos = self._target_states[..., 0:3]
        tar_rot = self._target_states[..., 3:7]

        pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]

        pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

        strike_distance = self.motions_dict[self.attack_type][self.movement_type]
        if self.movement_type == MovementType.RUN:
            dist_mask = pos_err < strike_distance
            self._tar_height[dist_mask] = MovementType.WALK.value
            strike_distance = self.motions_dict[self.attack_type][MovementType.WALK]

        dist_mask = pos_err < strike_distance
        self._should_strike[dist_mask] = 1
        dist_mask = pos_err >= strike_distance
        self._should_strike[dist_mask] = 0

        up = torch.zeros_like(tar_pos)
        up[..., -1] = 1
        tar_up = quat_rotate(tar_rot, up)
        tar_rot_err = torch.sum(up * tar_up, dim=-1)
        succ = tar_rot_err < 0.7

        self._stay_idle[succ] = 1

        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_states = self._target_states
            tar_height = self._tar_height
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
            tar_height = self._tar_height[env_ids]

        obs = compute_strike_heading_observations(root_states, tar_states, tar_height)

        return obs

    def _compute_reward(self, actions):
        tar_pos = self._target_states[..., 0:3]
        tar_rot = self._target_states[..., 3:7]
        char_root_state = self._humanoid_root_states
        strike_body_vel = self._rigid_body_vel[..., self._strike_body_ids[0], :]

        self.rew_buf[:] = compute_strike_reward(tar_pos, tar_rot, char_root_state,
                                                self._prev_root_pos, strike_body_vel,
                                                self.dt, self._near_dist)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces, self._contact_body_ids,
                                                                           self._rigid_body_pos,
                                                                           self._tar_contact_forces,
                                                                           self._strike_body_ids,
                                                                           self.max_episode_length,
                                                                           self._enable_early_termination,
                                                                           self._termination_heights)
        return

    def _draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._target_states[..., 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_strike_heading_observations(root_states, tar_pos, tar_height):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_vec = tar_pos[..., 0:2] - root_pos[..., 0:2]

    tar_dir = torch.nn.functional.normalize(tar_vec, dim=-1)
    heading_theta = torch.atan2(tar_dir[..., 1], tar_dir[..., 0])
    tar_dir = torch.stack([torch.cos(heading_theta), torch.sin(heading_theta)], dim=-1)

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_dir = quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]

    obs = torch.cat([local_tar_dir, tar_height.view(-1, 1)], dim=-1)
    return obs


@torch.jit.script
def compute_strike_reward(tar_pos, tar_rot, root_state, prev_root_pos, strike_body_vel, dt, near_dist):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    tar_speed = 1.0
    vel_err_scale = 4.0

    tar_rot_w = 0.6
    vel_reward_w = 0.4

    up = torch.zeros_like(tar_pos)
    up[..., -1] = 1
    tar_up = quat_rotate(tar_rot, up)
    tar_rot_err = torch.sum(up * tar_up, dim=-1)
    tar_rot_r = torch.clamp_min(1.0 - tar_rot_err, 0.0)

    root_pos = root_state[..., 0:3]
    tar_dir = tar_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    reward = tar_rot_w * tar_rot_r + vel_reward_w * vel_reward

    succ = tar_rot_err < 0.2
    reward = torch.where(succ, torch.ones_like(reward), reward)

    return reward


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           tar_contact_forces, strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 1.0

    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        tar_has_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)

        nonstrike_body_force = masked_contact_buf
        nonstrike_body_force[:, strike_body_ids, :] = 0
        nonstrike_body_has_contact = torch.any(torch.abs(nonstrike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)

        tar_fail = torch.logical_and(tar_has_contact, nonstrike_body_has_contact)

        has_failed = torch.logical_or(has_fallen, tar_fail)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
