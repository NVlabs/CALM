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

from learning.hrl_players import HRLPlayer


class HRLFSMPlayer(HRLPlayer):
    def _build_llc(self, config_params, checkpoint_file):
        super()._build_llc(config_params, checkpoint_file)

        self.env.task._possible_latents = self._get_motion_encoding()

    def _get_motion_encoding(self):
        all_encoded_demo_amp_obs = []
        for motion_id in range(self.env.task._motion_lib._motion_weights.shape[0]):
            motion_len = self.env.task._motion_lib._motion_lengths[motion_id]
            motion_time = torch.clamp(
                torch.clamp(motion_len - self.env.task.dt * self.env.task._num_amp_obs_enc_steps, min=0) / 2 + self.env.task.dt * self.env.task._num_amp_obs_enc_steps,
                max=motion_len.item()
            )

            motion_amp_obs = self.env.task.build_amp_obs_demo(
                torch.tensor([motion_id], device=self.device),
                motion_time,
                self.env.task._num_amp_obs_enc_steps
            ).view(-1, self.env.task._num_amp_obs_enc_steps, self.env.task._num_amp_obs_per_step)

            preproc_amp_obs = self._llc_agent._preproc_amp_obs(motion_amp_obs)
            encoded_demo_amp_obs = self._llc_agent.model.a2c_network.eval_enc(preproc_amp_obs)
            all_encoded_demo_amp_obs.append(encoded_demo_amp_obs)
        all_encoded_demo_amp_obs = torch.cat(all_encoded_demo_amp_obs, dim=0)

        return all_encoded_demo_amp_obs

    def env_step(self, env, obs_dict, action):
        requested_behavior = env.task.movement_type.value
        if self.env.task._should_strike[0].item() == 1:
            requested_behavior = env.task._strike_index
        if self.env.task._stay_idle[0].item() == 1:
            requested_behavior = env.task._idle_index

        string_behavior = env.task._motion_lib.motion_files[requested_behavior].split('Avatar_')[-1].split('_Motion')[0]
        print(f'Requested behavior: {string_behavior}')
        return super().env_step(env, obs_dict, action)

    def get_action(self, obs_dict, is_determenistic=False):
        obs = obs_dict['obs']

        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        proc_obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': proc_obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        current_action = torch.squeeze(current_action.detach())

        clamped_actions = torch.clamp(current_action, -1.0, 1.0)

        clamped_actions[self.env.task._should_strike == 1] = self.env.task._possible_latents[self.env.task._strike_index]
        clamped_actions[self.env.task._stay_idle == 1] = self.env.task._possible_latents[self.env.task._idle_index]

        return clamped_actions
