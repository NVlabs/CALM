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
from rl_games.algos_torch import players

from learning import amp_players

import numpy as np


class CALMPlayer(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        self._latent_dim = config['latent_dim']
        self._latent_steps_min = config.get('latent_steps_min', np.inf)
        self._latent_steps_max = config.get('latent_steps_max', np.inf)
        self._sample_latent_only_on_reset = config.get('latents_on_reset', None)  # if interpolate latents -- this should be set to false
        self._interpolate_latents = config.get('interpolate_latents', None)
        self._get_latents_from_data = config.get('latents_from_data', None)

        self._conditional_disc_reward_scale = config['conditional_disc_reward_scale']
        self._amp_batch_size = int(config['amp_batch_size'])

        if 'env' in config:
            self.env = config['env']

        super().__init__(config)

        if hasattr(self, 'env'):
            batch_size = self.env.task.num_envs
        else:
            batch_size = self.env_info['num_envs']
        self._calm_latents = torch.zeros((batch_size, self._latent_dim), dtype=torch.float32, device=self.device)

        if self._interpolate_latents is True:
            self._latents0 = torch.zeros((batch_size, self._latent_dim), dtype=torch.float32, device=self.device)
            self._latents1 = torch.zeros((batch_size, self._latent_dim), dtype=torch.float32, device=self.device)

            self._interpolation_alpha = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)

            self._interpolation_steps = 100
            self.env.task.max_episode_length = 500

        return

    def run(self):
        self._reset_latent_step_count()

        self.model.a2c_network._env = self.env
        super().run()
        return

    def _fetch_amp_obs_demo(self, num_samples):
        motion_ids, _, enc_amp_obs_demo_flat, _, _ = self.env.fetch_amp_obs_demo_enc_pair(num_samples)
        return motion_ids, enc_amp_obs_demo_flat

    def get_action(self, obs_dict, is_determenistic=False):
        if not self._sample_latent_only_on_reset:
            self._update_latents()

        obs = obs_dict['obs']
        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        obs = self._preproc_obs(obs)
        calm_latents = self._calm_latents

        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states,
            'calm_latents': calm_latents
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
        return players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))

    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)

        self._reset_latents(env_ids)
        return obs

    def _build_net_config(self):
        config = super()._build_net_config()
        config['calm_latent_shape'] = (self._latent_dim,)

        return config

    def _reset_latents(self, done_env_ids=None):
        if done_env_ids is None:
            num_envs = self.env.task.num_envs
            done_env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)

        num_samples = len(done_env_ids)

        if num_samples > 0:
            if self._interpolate_latents:
                num_samples *= 2

            if self._get_latents_from_data:
                z, motion_ids = self._sample_latents(num_samples)
            else:
                z = torch.normal(torch.zeros([num_samples, self._latent_dim], device=self.device))

            z = torch.nn.functional.normalize(z, dim=-1)

            if self._interpolate_latents:
                z, z1 = torch.chunk(z, 2)
                self._latents0[done_env_ids] = z
                self._latents1[done_env_ids] = z1
                self._interpolation_alpha[done_env_ids] = 1.

            self._calm_latents[done_env_ids] = z
            self._change_char_color(done_env_ids)

        return

    def _sample_latents(self, n):
        # 1. sample motion id
        # 2. create obs from entire sequence
        # 3. encode

        motion_ids, new_enc_amp_obs_demo = self._fetch_amp_obs_demo(n)

        proc_new_enc_amp_obs = self._preproc_amp_obs(new_enc_amp_obs_demo)
        with torch.no_grad():
            encoded_demo_amp_obs = self.model.a2c_network.eval_enc(proc_new_enc_amp_obs)

        # if we're interpolating from data, let's make it visually appealing by forcing the main character to have
        # two different motion files to interpolate from.
        motion_indices = motion_ids
        if self._interpolate_latents and motion_ids[0].item() == motion_ids[n // 2].item():
            for idx in range(1, n):
                if motion_ids[idx].item() != motion_ids[0].item():
                    encoded_demo_amp_obs[n // 2] = encoded_demo_amp_obs[idx]
                    break

        return encoded_demo_amp_obs, motion_indices

    def _update_latents(self):
        if self._latent_step_count <= 0:
            if self._interpolate_latents:
                self._interpolation_alpha -= 1. / self._interpolation_steps

                clamped_alpha = torch.clamp_min(self._interpolation_alpha, min=0)
                self._calm_latents = torch.nn.functional.normalize(self._latents0 * clamped_alpha + self._latents1 * (1 - clamped_alpha), dim=1)
            else:
                self._reset_latents()
            self._reset_latent_step_count()

            if self.env.task.viewer:
                print("Sampling new calm latents------------------------------")
                num_envs = self.env.task.num_envs
                env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)
                self._change_char_color(env_ids)
        else:
            self._latent_step_count -= 1
        return

    def _reset_latent_step_count(self):
        if self._interpolate_latents:
            self._latent_step_count = self.env.task.max_episode_length // self._interpolation_steps
        else:
            self._latent_step_count = np.random.randint(self._latent_steps_min, self._latent_steps_max)
        return

    def _calc_amp_rewards(self, amp_obs, calm_latents):
        disc_r = self._calc_disc_rewards(amp_obs)
        cdisc_r = self._calc_conditional_disc_rewards(amp_obs, calm_latents)
        output = {
            'disc_rewards': disc_r,
            'conditional_disc_rewards': cdisc_r
        }
        return output

    def _calc_conditional_disc_rewards(self, amp_obs, calm_latents):
        with torch.no_grad():
            disc_logits = self._eval_conditional_disc(amp_obs, calm_latents)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._conditional_disc_reward_scale

        return disc_r

    def _eval_conditional_disc(self, amp_obs, calm_latents):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_conditional_disc(proc_amp_obs, calm_latents)

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs
            calm_latents = self._calm_latents
            disc_pred = self._eval_conditional_disc(amp_obs, calm_latents)
            amp_rewards = self._calc_amp_rewards(amp_obs, calm_latents)
            disc_reward = amp_rewards['disc_rewards']
            cdisc_reward = amp_rewards['conditional_disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            cdisc_reward = cdisc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward, cdisc_reward)
        return

    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.env.task.set_char_color(rand_col, env_ids)
        return
