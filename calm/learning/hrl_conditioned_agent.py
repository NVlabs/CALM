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

import numpy as np

import torch
from torch.nn.functional import cosine_similarity

from learning.hrl_agent import HRLAgent


class HRLConditionedAgent(HRLAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        return

    def _build_llc(self, config_params, checkpoint_file):
        super()._build_llc(config_params, checkpoint_file)

        self.encoded_motion = self._get_motion_encoding()

    def _get_motion_encoding(self):
        all_encoded_demo_amp_obs = []
        for motion_id in range(self.vec_env.env.task._motion_lib._motion_weights.shape[0]):
            motion_amp_obs = self.vec_env.env.task.fetch_amp_obs_demo_per_id(32, motion_id)[-1].view(32, self.vec_env.env.task._num_amp_obs_enc_steps, self.vec_env.env.task._num_amp_obs_per_step)

            preproc_amp_obs = self._llc_agent._preproc_amp_obs(motion_amp_obs)
            encoded_demo_amp_obs = self._llc_agent.model.a2c_network.eval_enc(preproc_amp_obs)
            all_encoded_demo_amp_obs.append(encoded_demo_amp_obs)
        all_encoded_demo_amp_obs = torch.stack(all_encoded_demo_amp_obs, dim=0)

        return all_encoded_demo_amp_obs

    def _calc_style_reward(self, action):
        requested_heights = self.vec_env.env.task.tar_locomotion_index.view(-1)
        style_reward = torch.zeros((action.shape[0]), device=self.ppo_device, dtype=torch.float32)
        z = torch.nn.functional.normalize(action, dim=-1)

        for motion_type in range(3):
            motion_mask = requested_heights == motion_type
            style_reward[motion_mask] = torch.max((cosine_similarity(z[motion_mask].view(-1, 1, self._latent_dim), self.encoded_motion[motion_type].view(1, -1, self._latent_dim), dim=-1) + 1) / 2, dim=1)[0]

        return style_reward.unsqueeze(-1)

    def _calc_disc_reward(self, amp_obs):
        requested_motion_indices = self.vec_env.env.task.tar_locomotion_index.view(-1)

        disc_reward = torch.zeros((amp_obs.shape[0], 1), device=self.ppo_device, dtype=torch.float32)

        latent_index = np.random.randint(low=0, high=32)

        for motion_type in range(3):  # TODO: currently hardcoded to 3 motions.
            motion_mask = requested_motion_indices == motion_type
            disc_reward[motion_mask] = self._llc_agent._calc_conditional_disc_rewards(amp_obs[motion_mask], self.encoded_motion[0][latent_index].view(1, self._latent_dim).expand(amp_obs.shape[0], -1)[motion_mask])

        return disc_reward
