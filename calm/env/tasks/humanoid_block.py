# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils


class HumanoidBlock(humanoid_amp_task.HumanoidAMPTask):
    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )

        self._proj_dist_min = 1.5
        self._proj_dist_max = 3
        self._proj_h_min = 0.25
        self._proj_h_max = 2
        self._proj_steps = 70
        self._proj_warmup_steps = 40
        self._proj_speed_min = 30
        self._proj_speed_max = 40
        assert self._proj_warmup_steps < self._proj_steps

        self._build_proj_tensors()

        block_body_names = cfg["env"]["blockBodyNames"]
        self._block_body_ids = self._build_body_ids_tensor(
            self.envs[0], self.humanoid_handles[0], block_body_names
        )

        tar_body_names = cfg["env"]["tarBodyNames"]
        self._tar_body_ids = self._build_body_ids_tensor(
            self.envs[0], self.humanoid_handles[0], tar_body_names
        )

        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 18
        return obs_size

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._proj_handles = []
        self._load_proj_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_proj(env_id, env_ptr)
        return

    def _load_proj_asset(self):
        asset_root = "calm/data/assets/mjcf/"
        asset_file = "block_projectile.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._proj_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        return

    def _build_proj(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0

        proj_handle = self.gym.create_actor(
            env_ptr, self._proj_asset, default_pose, "proj", env_id, 2
        )
        self._proj_handles.append(proj_handle)

        return

    def _build_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name
            )
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_proj_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._proj_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., 1, :]

        self._proj_actor_ids = (
            to_torch(
                num_actors * np.arange(self.num_envs),
                device=self.device,
                dtype=torch.int32,
            )
            + 1
        )

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._proj_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., self.num_bodies, :]

        self._rel_proj_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )

        self._proj_hit_flag = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int
        )

        return

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_proj(env_ids)
        return

    def _reset_proj(self, env_ids):
        n = len(env_ids)

        rand_theta = (
            2
            * np.pi
            * torch.rand(
                [n], dtype=self._proj_states.dtype, device=self._proj_states.device
            )
        )
        rand_dist = (self._proj_dist_max - self._proj_dist_min) * torch.rand(
            [n], dtype=self._proj_states.dtype, device=self._proj_states.device
        ) + self._proj_dist_min
        rand_h = (self._proj_h_max - self._proj_h_min) * torch.rand(
            [n], dtype=self._proj_states.dtype, device=self._proj_states.device
        ) + self._proj_h_min

        self._rel_proj_pos[env_ids, 0] = rand_dist * torch.cos(rand_theta)
        self._rel_proj_pos[env_ids, 1] = rand_dist * torch.sin(rand_theta)
        self._rel_proj_pos[env_ids, 2] = rand_h

        self._proj_hit_flag[env_ids] = 0

        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._proj_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        return

    def _compute_task_obs(self, env_ids=None):
        proj_progress = self._calc_proj_progress()
        if env_ids is None:
            root_states = self._humanoid_root_states
            proj_states = self._proj_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            proj_states = self._proj_states[env_ids]
            proj_progress = proj_progress[env_ids]

        proj_phase = proj_progress.float() / self._proj_steps
        obs = compute_block_observations(root_states, proj_phase, proj_states)
        return obs

    def _compute_reward(self, actions):
        char_root_pos = self._humanoid_root_states[..., 0:3]
        self.rew_buf[:] = compute_block_reward(char_root_pos, self._proj_hit_flag)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            self._contact_forces,
            self._contact_body_ids,
            self._humanoid_root_states,
            self._rigid_body_pos,
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_heights,
            self._proj_hit_flag,
        )
        return

    def _physics_step(self):
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            self._update_proj_hit_buffers()

        return

    def post_physics_step(self):
        self._update_proj()
        super().post_physics_step()

        return

    def _update_proj_hit_buffers(self):
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self._proj_hit_flag[:] = compute_proj_hit_buffer(
            self._contact_forces,
            self._proj_contact_forces,
            self._contact_body_ids,
            self._block_body_ids,
            self._proj_hit_flag,
        )
        return

    def _update_proj(self):
        humanoid_root_pos = self._humanoid_root_states[..., 0:3]

        proj_progress = self._calc_proj_progress()
        reset_mask = proj_progress == 0
        reset_envs = torch.nonzero(reset_mask, as_tuple=False)
        self._reset_proj(reset_envs[:, 0])

        warmup_mask = proj_progress < self._proj_warmup_steps
        self._proj_states[warmup_mask, 0:2] = (
            humanoid_root_pos[warmup_mask, 0:2] + self._rel_proj_pos[warmup_mask, 0:2]
        )
        self._proj_states[warmup_mask, 2] = self._rel_proj_pos[warmup_mask, 2]
        self._proj_states[warmup_mask, 3:6] = 0.0
        self._proj_states[warmup_mask, 6] = 1.0
        self._proj_states[warmup_mask, 7:10] = 0.0
        self._proj_states[warmup_mask, 10:13] = 0.0

        launch_mask = proj_progress == self._proj_warmup_steps
        launch_pos = self._proj_states[launch_mask, 0:3]

        num_tar_bodies = self._tar_body_ids.shape[0]
        rand_tar_idx = torch.randint(num_tar_bodies, [launch_pos.shape[0]])
        tar_body_idx = self._tar_body_ids[rand_tar_idx]

        launch_tar_pos = self._rigid_body_pos[launch_mask, tar_body_idx, :]
        launch_dir = launch_tar_pos - launch_pos
        launch_dir += 0.1 * torch.randn_like(launch_dir)
        launch_dir = torch.nn.functional.normalize(launch_dir, dim=-1)
        launch_speed = (self._proj_speed_max - self._proj_speed_min) * torch.rand_like(
            launch_dir[:, 0:1]
        ) + self._proj_speed_min
        launch_vel = launch_speed * launch_dir
        self._proj_states[launch_mask, 7:10] = launch_vel

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._proj_actor_ids),
            len(self._proj_actor_ids),
        )

        if self.viewer:
            self._update_proj_color()

        return

    def _update_proj_color(self):
        col0 = to_torch(
            np.array([0.7, 0.8, 0.9]), device=self.device, dtype=torch.float
        )
        col1 = to_torch(
            np.array([1.0, 0.3, 0.3]), device=self.device, dtype=torch.float
        )

        proj_progress = self._calc_proj_progress()
        warmup_phase = proj_progress.float() / self._proj_warmup_steps
        warmup_phase = torch.clamp_max(warmup_phase, 1.0)

        col_lerp = warmup_phase * warmup_phase * (3.0 - 2.0 * warmup_phase)
        col_lerp = col_lerp.unsqueeze(-1)
        cols = (1 - col_lerp) * col0 + col_lerp * col1

        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self._proj_handles[i]

            curr_col = cols[0]

            self.gym.set_rigid_body_color(
                env_ptr,
                handle,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(curr_col[0], curr_col[1], curr_col[2]),
            )

        return

    def _calc_proj_progress(self):
        proj_progress = torch.remainder(self.progress_buf, self._proj_steps)
        return proj_progress

    def _draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._proj_states[..., 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(
                self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols
            )

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_block_observations(root_states, proj_phase, proj_states):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    proj_pos = proj_states[:, 0:3]
    proj_rot = proj_states[:, 3:7]
    proj_vel = proj_states[:, 7:10]
    proj_ang_vel = proj_states[:, 10:13]

    tar_pos = 0
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    local_tar_pos = quat_rotate(heading_rot, tar_pos - root_pos)
    local_tar_pos = local_tar_pos[..., 0:2]

    local_proj_pos = proj_pos - root_pos
    local_proj_pos = quat_rotate(heading_rot, local_proj_pos)
    local_proj_vel = quat_rotate(heading_rot, proj_vel)
    local_proj_ang_vel = quat_rotate(heading_rot, proj_ang_vel)

    local_proj_rot = quat_mul(heading_rot, proj_rot)
    local_proj_rot_obs = torch_utils.quat_to_tan_norm(local_proj_rot)

    proj_phase = proj_phase.unsqueeze(-1)

    obs = torch.cat(
        [
            proj_phase,
            local_tar_pos,
            local_proj_pos,
            local_proj_rot_obs,
            local_proj_vel,
            local_proj_ang_vel,
        ],
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_block_reward(root_pos, proj_hit_flag):
    # type: (Tensor, Tensor) -> Tensor
    pos_err_scale = 0.5

    pos_reward_w = 0.0
    hit_reward_w = 1.0

    pos_diff = root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    hit_reward = torch.zeros_like(pos_reward)
    hit_reward[proj_hit_flag == 1] = 1.0

    reward = pos_reward_w * pos_reward + hit_reward_w * hit_reward

    reward[proj_hit_flag == -1] = 0.0

    return reward


@torch.jit.script
def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    contact_body_ids,
    root_states,
    rigid_body_pos,
    max_episode_length,
    enable_early_termination,
    termination_heights,
    proj_hit_flag,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    tar_fail_dist_threshold = 2.0

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

        proj_fail = proj_hit_flag == -1

        root_pos = root_states[..., 0:3]
        tar_dist_sq = (
            root_pos[..., 0] * root_pos[..., 0] + root_pos[..., 1] * root_pos[..., 1]
        )
        tar_dist_fail = tar_dist_sq > tar_fail_dist_threshold * tar_fail_dist_threshold

        has_failed = torch.logical_or(has_fallen, proj_fail)
        has_failed = torch.logical_or(has_failed, tar_dist_fail)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= progress_buf > 1
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated


@torch.jit.script
def compute_proj_hit_buffer(
    contact_buf, proj_contact_buf, contact_body_ids, block_body_ids, proj_hit_buffer
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    contact_force_threshold = 1.0

    block_contact_buf = contact_buf[:, block_body_ids, :]
    block_body_has_contact = torch.any(
        torch.abs(block_contact_buf) > contact_force_threshold, dim=-1
    )
    block_body_has_contact = torch.any(block_body_has_contact, dim=-1)

    nonblock_contact_buf = contact_buf.clone()
    nonblock_contact_buf[:, contact_body_ids, :] = 0
    nonblock_contact_buf[:, block_body_ids, :] = 0
    nonblock_body_has_contact = torch.any(
        torch.abs(nonblock_contact_buf) > contact_force_threshold, dim=-1
    )
    nonblock_body_has_contact = torch.any(nonblock_body_has_contact, dim=-1)

    proj_has_contact = torch.any(
        torch.abs(proj_contact_buf) > contact_force_threshold, dim=-1
    )

    first_hit = proj_hit_buffer == 0
    hit_fail = torch.logical_and(proj_has_contact, nonblock_body_has_contact)
    hit_fail = torch.logical_and(hit_fail, first_hit)

    hit_succ = torch.logical_and(proj_has_contact, block_body_has_contact)
    hit_succ = torch.logical_and(hit_succ, torch.logical_not(nonblock_body_has_contact))
    hit_succ = torch.logical_and(hit_succ, first_hit)

    proj_hit_buffer[hit_fail] = -1
    proj_hit_buffer[hit_succ] = 1

    return proj_hit_buffer
