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

from learning import amp_models


class ModelCALMContinuous(amp_models.ModelAMPContinuous):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('calm', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelCALMContinuous.Network(net)

    class Network(amp_models.ModelAMPContinuous.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            result = super().forward(input_dict)

            if is_train:
                amp_obs = input_dict['amp_obs']
                amp_demo_obs = input_dict['amp_obs_demo']
                amp_obs_replay = input_dict['amp_obs_replay']

                batched_calm_latents = input_dict['batched_calm_latents']
                conditional_disc_agent_logit = self.a2c_network.eval_conditional_disc(amp_obs, batched_calm_latents)
                result["conditional_disc_agent_logit"] = conditional_disc_agent_logit

                calm_latents_demo = input_dict['calm_latents_demo']
                conditional_disc_agent_logit = self.a2c_network.eval_conditional_disc(amp_demo_obs, calm_latents_demo)
                result["conditional_disc_demo_logit"] = conditional_disc_agent_logit

                calm_latents_replay = input_dict['calm_latents_replay']
                conditional_disc_agent_replay_logit = self.a2c_network.eval_conditional_disc(amp_obs_replay, calm_latents_replay)
                result["conditional_disc_agent_replay_logit"] = conditional_disc_agent_replay_logit

                conditional_disc_neg_demo_logit = self.a2c_network.eval_conditional_disc(amp_demo_obs, batched_calm_latents)
                result["conditional_disc_neg_demo_logit"] = conditional_disc_neg_demo_logit

            return result
