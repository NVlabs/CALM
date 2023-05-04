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

from learning import amp_agent 

import torch

import numpy as np
from isaacgym.torch_utils import *
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common

import time


class CALMAgent(amp_agent.AMPAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        self.model.a2c_network._env = self.vec_env.env

        return

    def init_tensors(self):
        super().init_tensors()
        
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['calm_latents'] = torch.zeros(batch_shape + (self._latent_dim,), dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['enc_amp_obs'] = torch.zeros(batch_shape + self._enc_amp_observation_space.shape, dtype=torch.float32, device=self.ppo_device)
        
        self._calm_latents = torch.zeros((batch_shape[-1], self._latent_dim), dtype=torch.float32, device=self.ppo_device)
        self._enc_amp_obs = torch.zeros((batch_shape[-1], self._enc_amp_observation_space.shape[-1]), dtype=torch.float32, device=self.ppo_device)
        
        self.tensor_list += ['calm_latents', 'enc_amp_obs']

        self._latent_reset_steps = torch.zeros(batch_shape[-1], dtype=torch.int32, device=self.ppo_device)
        num_envs = self.vec_env.env.task.num_envs
        env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.ppo_device)
        self._reset_latent_step_count(env_ids)

        return

    def play_steps(self):
        self.set_eval()
        
        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            self._update_latents()

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, self._calm_latents, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._calm_latents, self._rand_action_probs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])
            self.experience_buffer.update_data('calm_latents', n, self._calm_latents)
            self.experience_buffer.update_data('enc_amp_obs', n, self._enc_amp_obs)
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs, self._calm_latents)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
            if self.vec_env.env.task.viewer:
                self._amp_debug(infos, self._calm_latents)

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        mb_calm_latents = self.experience_buffer.tensor_dict['calm_latents']
        amp_rewards = self._calc_amp_rewards(mb_amp_obs, mb_calm_latents)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict

    def get_action_values(self, obs_dict, calm_latents, rand_action_probs):
        processed_obs = self._preproc_obs(obs_dict['obs'])

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs': processed_obs,
            'rnn_states': self.rnn_states,
            'calm_latents': calm_latents
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs_dict['states']
                input_dict = {
                    'is_train': False,
                    'states': states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        
        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]
        res_dict['rand_action_mask'] = rand_action_mask

        return res_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)

        self.dataset.values_dict['enc_amp_obs'] = batch_dict['enc_amp_obs']
        self.dataset.values_dict['enc_amp_obs_replay'] = batch_dict['enc_amp_obs_replay']
        self.dataset.values_dict['enc_amp_obs_demo'] = batch_dict['enc_amp_obs_demo']

        return

    def train_epoch(self):
        play_time_start = time.time()

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self._update_amp_demos()
        num_obs_samples = batch_dict['amp_obs'].shape[0]
        samples = self._amp_obs_demo_buffer.sample(num_obs_samples)
        batch_dict['amp_obs_demo'] = samples['amp_obs']
        batch_dict['enc_amp_obs_demo'] = samples['enc_amp_obs']

        if self._amp_replay_buffer.get_total_count() == 0:
            batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
            batch_dict['enc_amp_obs_replay'] = batch_dict['enc_amp_obs']
        else:
            samples = self._amp_replay_buffer.sample(num_obs_samples)
            batch_dict['amp_obs_replay'] = samples['amp_obs']
            batch_dict['enc_amp_obs_replay'] = samples['enc_amp_obs']

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef,
                                                                            self.epoch_num, 0,
                                                                            curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if train_info is None:
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num,
                                                                        0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,
                                                                    av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        self._store_replay_amp_obs(batch_dict['amp_obs'], batch_dict['enc_amp_obs'])

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        enc_amp_obs = self._preproc_amp_obs(input_dict['enc_amp_obs'])

        mb_enc_amp_obs_demo = input_dict['enc_amp_obs_demo'][0:self._amp_minibatch_size]
        mb_enc_amp_obs_demo = self._preproc_amp_obs(mb_enc_amp_obs_demo)

        mb_amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        mb_amp_obs = self._preproc_amp_obs(mb_amp_obs)

        mb_amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        mb_amp_obs_replay = self._preproc_amp_obs(mb_amp_obs_replay)

        mb_amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        mb_amp_obs_demo = self._preproc_amp_obs(mb_amp_obs_demo)
        mb_amp_obs_demo.requires_grad_(True)

        with torch.no_grad():
            mb_calm_latents_demo = self._eval_enc(mb_enc_amp_obs_demo)
        mb_calm_latents_demo.requires_grad_(True)

        # Update relevant latents with output from enc to drive gradients backward from policy to the encoder.
        amp_obs_encoding = self._eval_enc(enc_amp_obs)
        calm_latents = amp_obs_encoding

        mb_calm_latents = calm_latents[0:self._amp_minibatch_size]

        mb_enc_amp_obs_replay = input_dict['enc_amp_obs_replay'][0:self._amp_minibatch_size]
        mb_enc_amp_obs_replay = self._preproc_amp_obs(mb_enc_amp_obs_replay)
        with torch.no_grad():
            mb_calm_latents_replay = self._eval_enc(mb_enc_amp_obs_replay)

        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)

        encoder_uniformity = uniform_loss(amp_obs_encoding.detach())

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'eval_disc': self._disc_reward_w > 0,
            'prev_actions': actions_batch,
            'obs': obs_batch,

            'amp_obs': mb_amp_obs,
            'amp_obs_replay': mb_amp_obs_replay,
            'amp_obs_demo': mb_amp_obs_demo,

            'calm_latents': calm_latents,
            'calm_latents_replay': mb_calm_latents_replay,
            'batched_calm_latents': mb_calm_latents.detach(),
            'calm_latents_demo': mb_calm_latents_demo
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)

            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            conditional_disc_agent_logit = res_dict['conditional_disc_agent_logit']
            conditional_disc_agent_replay_logit = res_dict['conditional_disc_agent_replay_logit']
            conditional_disc_demo_logit = res_dict['conditional_disc_demo_logit']
            conditional_disc_negative_demo_logit = res_dict['conditional_disc_neg_demo_logit']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            a_clipped = a_info['actor_clipped'].float()

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)

            c_loss = torch.mean(c_loss)
            a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
            entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
            b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
            a_clip_frac = torch.sum(rand_action_mask * a_clipped) / rand_action_sum

            if self._disc_reward_w <= 0:
                disc_loss = 0
            else:
                disc_agent_logit = res_dict['disc_agent_logit']
                disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
                disc_demo_logit = res_dict['disc_demo_logit']
                disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
                disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, mb_amp_obs_demo)
                disc_loss = disc_info['disc_loss']

            # Regularization for the learned encodings
            if self._enc_reg_coeff <= 0:
                enc_reg_loss = 0
            else:
                enc_reg_info = self._enc_reg_loss()
                enc_reg_loss = enc_reg_info['enc_reg_loss']

            # Regularization for the discriminator
            if self._negative_disc_samples:
                conditional_disc_agent_cat_logit = torch.cat([conditional_disc_agent_logit, conditional_disc_agent_replay_logit, conditional_disc_negative_demo_logit], dim=0)
            else:
                conditional_disc_agent_cat_logit = torch.cat([conditional_disc_agent_logit, conditional_disc_agent_replay_logit], dim=0)

            conditional_disc_info = self._conditional_disc_loss(conditional_disc_agent_cat_logit, conditional_disc_demo_logit, mb_amp_obs_demo, mb_calm_latents_demo)
            conditional_disc_loss = conditional_disc_info['conditional_disc_loss']

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                   + self._disc_coef * disc_loss + self._conditional_disc_coef * conditional_disc_loss + self._enc_reg_coeff * enc_reg_loss

            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
        
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss,
            'encoder_uniformity': encoder_uniformity
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        if self._disc_reward_w > 0:
            self.train_result.update(disc_info)
        self.train_result.update(conditional_disc_info)

        return

    def _conditional_disc_loss(self, disc_agent_logit, disc_demo_logit, obs_hrl, calm_latents):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_conditional_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._conditional_disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, (obs_hrl, calm_latents),
                                             grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if self._conditional_disc_weight_decay != 0:
            disc_weights = self.model.a2c_network.get_conditional_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._conditional_disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            'conditional_disc_loss': disc_loss,
            'conditional_disc_grad_penalty': disc_grad_penalty.detach(),
            'conditional_disc_logit_loss': disc_logit_loss.detach(),
            'conditional_disc_agent_acc': disc_agent_acc.detach(),
            'conditional_disc_demo_acc': disc_demo_acc.detach(),
            'conditional_disc_agent_logit': disc_agent_logit.detach(),
            'conditional_disc_demo_logit': disc_demo_logit.detach()
        }
        return disc_info

    def _enc_reg_loss(self):
        enc_amp_obs_demo, _ = self._fetch_amp_obs_demo(self._amp_minibatch_size)
        proc_enc_amp_obs_demo = self._preproc_amp_obs(enc_amp_obs_demo)

        amp_obs_encoding = self._eval_enc(proc_enc_amp_obs_demo)

        # Loss for uniform distribution over the sphere
        uniform_l = uniform_loss(amp_obs_encoding)

        _, _, similar_enc_amp_obs_demo0, _, similar_enc_amp_obs_demo1 = self.vec_env.env.task.fetch_amp_obs_demo_pair(self._amp_minibatch_size)

        proc_similar_enc_amp_obs_demo0 = self._preproc_amp_obs(similar_enc_amp_obs_demo0)
        proc_similar_enc_amp_obs_demo1 = self._preproc_amp_obs(similar_enc_amp_obs_demo1)

        similar_amp_obs_encoding0 = self._eval_enc(proc_similar_enc_amp_obs_demo0)
        similar_amp_obs_encoding1 = self._eval_enc(proc_similar_enc_amp_obs_demo1)

        # Loss for alignment - overlapping motions should have 'close' embeddings
        align_l = align_loss(similar_amp_obs_encoding0, similar_amp_obs_encoding1)

        loss = align_l + 0.5 * uniform_l

        return {'enc_reg_loss': loss}

    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)
        
        if env_ids is None:
            num_envs = self.vec_env.env.task.num_envs
            env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.ppo_device)

        if len(env_ids) > 0:
            self._reset_latents(env_ids)
            self._reset_latent_step_count(env_ids)

        return obs

    def _reset_latent_step_count(self, env_ids):
        self._latent_reset_steps[env_ids] = torch.randint_like(self._latent_reset_steps[env_ids], low=self._latent_steps_min,
                                                               high=self._latent_steps_max)
        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._latent_dim = config['latent_dim']
        self._latent_steps_min = config.get('latent_steps_min', np.inf)
        self._latent_steps_max = config.get('latent_steps_max', np.inf)

        self._conditional_disc_logit_reg = config['conditional_disc_logit_reg']
        self._conditional_disc_grad_penalty = config['conditional_disc_grad_penalty']
        self._conditional_disc_weight_decay = config['conditional_disc_weight_decay']
        self._conditional_disc_reward_scale = config['conditional_disc_reward_scale']

        self._conditional_disc_coef = config['conditional_disc_coef']
        self._conditional_disc_reward_w = config['conditional_disc_reward_w']
        self._conditional_disc_reward_scale = config['conditional_disc_reward_scale']

        self._negative_disc_samples = config.get('negative_disc_samples', False)

        self._enc_reg_coeff = config.get('enc_regularization_coeff', 0)

        self._enc_amp_observation_space = self.env_info['enc_amp_observation_space']

        if not hasattr(self, 'vec_env'):
            self.vec_env = config.get('vec_env')

        return
    
    def _build_net_config(self):
        config = super()._build_net_config()
        config['calm_latent_shape'] = (self._latent_dim,)
        return config

    def _reset_latents(self, env_ids):
        n = len(env_ids)
        z, enc_amp_obs_demo = self._sample_latents(n)
        self._calm_latents[env_ids] = z
        self._enc_amp_obs[env_ids] = enc_amp_obs_demo

        if self.vec_env.env.task.viewer:
            self._change_char_color(env_ids)

        return

    def _sample_latents(self, n):
        enc_amp_obs_demo, _ = self._fetch_amp_obs_demo(n)
        with torch.no_grad():
            proc_enc_amp_obs_demo = self._preproc_amp_obs(enc_amp_obs_demo)
            latents = self.model.a2c_network.eval_enc(proc_enc_amp_obs_demo)

        return latents, enc_amp_obs_demo

    def _update_latents(self):
        new_latent_envs = self._latent_reset_steps <= self.vec_env.env.task.progress_buf

        need_update = torch.any(new_latent_envs)
        if need_update:
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            self._reset_latents(new_latent_env_ids)
            self._latent_reset_steps[new_latent_env_ids] += torch.randint_like(self._latent_reset_steps[new_latent_env_ids],
                                                                               low=self._latent_steps_min, 
                                                                               high=self._latent_steps_max)
            if self.vec_env.env.task.viewer:
                self._change_char_color(new_latent_env_ids)

        return

    def _eval_actor(self, obs, calm_latents):
        output = self.model.a2c_network.eval_actor(obs=obs, calm_latents=calm_latents)
        return output

    def _eval_enc(self, amp_obs):
        output = self.model.a2c_network.eval_enc(amp_obs=amp_obs)
        return output

    def _eval_critic(self, obs_dict, calm_latents):
        self.model.eval()
        obs = obs_dict['obs']
        processed_obs = self._preproc_obs(obs)
        value = self.model.a2c_network.eval_critic(processed_obs, calm_latents)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _calc_amp_rewards(self, amp_obs, calm_latents):
        cdisc_r = self._calc_conditional_disc_rewards(amp_obs, calm_latents)
        if self._disc_reward_w <= 0:
            disc_r = torch.zeros_like(cdisc_r)
        else:
            disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r,
            'conditional_disc_rewards': cdisc_r
        }
        return output

    def _calc_conditional_disc_rewards(self, amp_obs, calm_latents):
        with torch.no_grad():
            disc_logits = self._eval_conditional_disc(amp_obs, calm_latents)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._conditional_disc_reward_scale

        return disc_r

    def _eval_conditional_disc(self, amp_obs_calm, calm_latents):
        proc_amp_obs_calm = self._preproc_amp_obs(amp_obs_calm)
        return self.model.a2c_network.eval_conditional_disc(proc_amp_obs_calm, calm_latents)

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']
        conditional_disc_r = amp_rewards['conditional_disc_rewards']
        combined_rewards = self._task_reward_w * task_rewards \
                         + self._disc_reward_w * disc_r \
                         + self._conditional_disc_reward_w * conditional_disc_r
        return combined_rewards

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['conditional_disc_rewards'] = batch_dict['conditional_disc_rewards']
        return

    def _store_replay_amp_obs(self, amp_obs, enc_amp_obs):
        if amp_obs.shape[0] > 0:
            buf_size = self._amp_replay_buffer.get_buffer_size()
            buf_total_count = self._amp_replay_buffer.get_total_count()
            if (buf_total_count > buf_size):
                keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
                keep_mask = torch.bernoulli(keep_probs) == 1.0
                amp_obs = amp_obs[keep_mask]
                enc_amp_obs = enc_amp_obs[keep_mask]

            if (amp_obs.shape[0] > buf_size):
                rand_idx = torch.randperm(amp_obs.shape[0])
                rand_idx = rand_idx[:buf_size]
                amp_obs = amp_obs[rand_idx]
                enc_amp_obs = enc_amp_obs[rand_idx]

            self._amp_replay_buffer.store({'amp_obs': amp_obs, 'enc_amp_obs': enc_amp_obs})
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        
        self.writer.add_scalar('losses/conditional_disc_loss', torch_ext.mean_list(train_info['conditional_disc_loss']).item(), frame)

        self.writer.add_scalar('info/conditional_disc_agent_acc', torch_ext.mean_list(train_info['conditional_disc_agent_acc']).item(), frame)
        self.writer.add_scalar('info/conditional_disc_demo_acc', torch_ext.mean_list(train_info['conditional_disc_demo_acc']).item(), frame)
        self.writer.add_scalar('info/conditional_disc_agent_logit', torch_ext.mean_list(train_info['conditional_disc_agent_logit']).item(), frame)
        self.writer.add_scalar('info/conditional_disc_demo_logit', torch_ext.mean_list(train_info['conditional_disc_demo_logit']).item(), frame)
        self.writer.add_scalar('info/conditional_disc_grad_penalty', torch_ext.mean_list(train_info['conditional_disc_grad_penalty']).item(), frame)
        self.writer.add_scalar('info/conditional_disc_logit_loss', torch_ext.mean_list(train_info['conditional_disc_logit_loss']).item(), frame)

        conditional_disc_reward_std, conditional_disc_reward_mean = torch.std_mean(train_info['conditional_disc_rewards'])
        self.writer.add_scalar('info/conditional_disc_reward_mean', conditional_disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/conditional_disc_reward_std', conditional_disc_reward_std.item(), frame)

        self.writer.add_scalar('info/encoder_uniformity', torch_ext.mean_list(train_info['encoder_uniformity']).item(), frame)

        return

    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.vec_env.env.task.set_char_color(rand_col, env_ids)
        return

    def _amp_debug(self, info, calm_latents):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs
            calm_latents = calm_latents
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs, calm_latents)
            disc_reward = amp_rewards['disc_rewards']
            cdisc_reward = amp_rewards['conditional_disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            cdisc_reward = cdisc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward, cdisc_reward)
        return

    def _fetch_amp_obs_demo(self, num_samples):
        _, _, enc_amp_obs_demo_flat, _, amp_obs_demo_flat = self.vec_env.env.fetch_amp_obs_demo_enc_pair(num_samples)
        return enc_amp_obs_demo_flat, amp_obs_demo_flat

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            enc_amp_obs_demo, amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': amp_obs_demo, 'enc_amp_obs': enc_amp_obs_demo})

        return

    def _update_amp_demos(self):
        enc_amp_obs_demo, amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': amp_obs_demo, 'enc_amp_obs': enc_amp_obs_demo})
        return


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def align_loss(x, y, alpha=2):
    return torch.linalg.norm(x - y, ord=2, dim=1).pow(alpha).mean()
