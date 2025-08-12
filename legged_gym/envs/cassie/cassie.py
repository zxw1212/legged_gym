# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi

class Cassie(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()

        self.ee_posi = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.ee_linear_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_angular_vel = torch.zeros((self.num_envs, 3), device=self.device)

        rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_states)

        # rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.asset)
        rigid_body_dict = self.gym.get_actor_rigid_body_dict(self.envs[0], self.actor_handles[0])
        target_ee_name = "iiwa_link1"
        if target_ee_name not in rigid_body_dict:
            raise KeyError(f"Link '{target_ee_name}' not found in actor rigid bodies. Available names: {list(rigid_body_dict.keys())}")
        self.ee_index = rigid_body_dict["iiwa_link1"]
        target_ee_joint_name = "iiwa_joint1"
        if target_ee_joint_name not in self.dof_names:
            raise KeyError(f"Joint '{target_ee_joint_name}' not found in actor DOFs. Available names: {self.dof_names}")
        self.ee_joint_index = self.dof_names.index("iiwa_joint1")

        rb_states_reshaped = self.rigid_body_states.view(self.num_envs, -1, 13)

        self.ee_posi[:] = rb_states_reshaped[:, self.ee_index, 0:3]
        self.ee_rot[:] = rb_states_reshaped[:, self.ee_index, 3:7]
        # ee vel in world frame
        self.ee_linear_vel[:] = rb_states_reshaped[:, self.ee_index, 7:10]
        self.ee_angular_vel[:] = rb_states_reshaped[:, self.ee_index, 10:13]
        # ee vel in local frame
        self.ee_linear_vel[:] = quat_rotate_inverse(self.base_quat, self.ee_linear_vel)
        self.ee_angular_vel[:] = quat_rotate_inverse(self.base_quat, self.ee_angular_vel)


    def post_physics_step(self):
        super().post_physics_step()

        # update ee state
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rb_states_reshaped = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.ee_posi[:] = rb_states_reshaped[:, self.ee_index, 0:3]
        self.ee_rot[:] = rb_states_reshaped[:, self.ee_index, 3:7]
        self.ee_linear_vel[:] = rb_states_reshaped[:, self.ee_index, 7:10]
        self.ee_angular_vel[:] = rb_states_reshaped[:, self.ee_index, 10:13]
        self.ee_linear_vel[:] = quat_rotate_inverse(self.base_quat, self.ee_linear_vel)
        self.ee_angular_vel[:] = quat_rotate_inverse(self.base_quat, self.ee_angular_vel)


    def compute_observations(self):
        # 复制base类的obs计算（注意你的self.obs_scales和self.commands_scale等配置）
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

        # 如果需要测高
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # # 拼接ee状态，13维, 使用ee joint的角度作和速度来计算ee heading也可以,减少了obs size, 减轻训练难度
        # extra = torch.cat((self.ee_posi, self.ee_rot, self.ee_linear_vel, self.ee_angular_vel), dim=-1)
        # self.obs_buf = torch.cat((self.obs_buf, extra), dim=-1)

        # 加噪声
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _reward_ee_heading(self):
        """
        使用ee joint做 ee heading 计算
        由于ee link是通过ee joint直接链接在base上的, 所以ee joint的数值即代表ee 相对 base 的yaw角度且符号相同.
        """
        # reward for ee heading
        # hardcode heading: yaw = 0.0, relative to base frame
        ee_yaw_in_base = self.dof_pos[:, self.ee_joint_index]
        # 计算ee相对于base的yaw误差，wrap到[-pi,pi]
        yaw_error = wrap_to_pi(ee_yaw_in_base)
        # 期望ee角速度绕z轴，比例控制器(比例系数0.5可调)
        desired_ee_angular_vel_z = torch.clamp(-0.5 * yaw_error, -1.0, 1.0)
        # 当前观测到的ee角速度的z分量（局部坐标系）, 其实就是ee joint的速度
        ee_ang_vel_z = self.dof_vel[:, self.ee_joint_index]
        # 计算角速度误差
        ang_vel_error = desired_ee_angular_vel_z - ee_ang_vel_z
        # 以误差平方的负指数作为奖励（误差越小，奖励越大）
        reward = torch.exp(-ang_vel_error**2 / self.cfg.rewards.tracking_sigma)
        return reward

    
    # def _reward_ee_heading(self):
    #     """
    #     使用ee pose做ee heading 计算.
    #     """
    #     # reward for ee heading
    #     # hardcode heading: yaw = 0.0, relative to base frame
    #     ee_yaw = torch.atan2(
    #         2*(self.ee_rot[:,3]*self.ee_rot[:,2] + self.ee_rot[:,0]*self.ee_rot[:,1]),  # sin yaw
    #         1 - 2*(self.ee_rot[:,1]**2 + self.ee_rot[:,2]**2)                        # cos yaw
    #     )

    #     # base四元数转yaw
    #     base_yaw = torch.atan2(
    #         2*(self.base_quat[:,3]*self.base_quat[:,2] + self.base_quat[:,0]*self.base_quat[:,1]),
    #         1 - 2*(self.base_quat[:,1]**2 + self.base_quat[:,2]**2)
    #     )

    #      # 计算ee相对于base的yaw误差，wrap到[-pi,pi]
    #     yaw_error = wrap_to_pi(ee_yaw - base_yaw)

    #     # 期望ee角速度绕z轴，比例控制器(比例系数0.5可调)
    #     desired_ee_angular_vel_z = torch.clamp(-0.5 * yaw_error, -1.0, 1.0)

    #     # 当前观测到的ee角速度的z分量（局部坐标系）
    #     ee_ang_vel_z = self.ee_angular_vel[:, 2]

    #     # 计算角速度误差
    #     ang_vel_error = desired_ee_angular_vel_z - ee_ang_vel_z

    #     # 以误差平方的负指数作为奖励（误差越小，奖励越大）
    #     reward = torch.exp(-ang_vel_error**2 / self.cfg.rewards.tracking_sigma)

    #     return reward