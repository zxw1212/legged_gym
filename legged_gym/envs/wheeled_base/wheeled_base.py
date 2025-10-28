from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi

class WheeledBase(LeggedRobot):
    def _init_buffers(self):
        """
        copy from legged_robot.py, anded staff is marked below with ==============
        """

        # ============================= addeded ==============================
        # define wheel joint names
        self.wheel_joint_names = [
            "wheel_RF_Joint",  # 右前
            "wheel_LF_Joint",  # 左前
            "wheel_RR_Joint",  # 右后
            "wheel_LR_Joint",  # 左后
        ]

        # 获取轮子关节的索引 
        # asset_dof_names = self.gym.get_asset_dof_names(self.asset)
        # self.wheel_joint_indices = [asset_dof_names.index(name) for name in self.wheel_joint_names]
        self.wheel_joint_indices = [self.dof_names.index(name) for name in self.wheel_joint_names]

        print("wheel joint indices:", self.wheel_joint_indices)
        # ============================= addeded ==============================


        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # 索引范围	数据含义
        # 0:3	位置 (x, y, z)
        # 3:7	旋转 (四元数 w, x, y, z)
        # 7:10	线速度 (vx, vy, vz)
        # 10:13	角速度 (wx, wy, wz)
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        # self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # ============================= addeded ==============================
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        # ============================= addeded ==============================


        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

            #========================== addeded ==============================
            # TODO: set dof properties from cfg
            props["lower"] = np.clip(props["lower"], -1e6, 1e6) # avoid upper-lower gettingg inf, inf*gain=nan
            props["upper"] = np.clip(props["upper"], -1e6, 1e6)
            props["velocity"] = np.clip(props["velocity"], -1e6, 1e6)
            props["effort"] = np.clip(props["effort"], -1e6, 1e6)
            props["stiffness"] *= 0.0
            #========================== addeded ==============================

            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()

                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

            
            # print("dof props:", type(props), props) # ndarray
            # print("props dtype:", props.dtype)## 'hasLimits', 'lower', 'upper', 'driveMode', 'velocity', 'effort', 'stiffness', 'damping', 'friction', 'armature'
            # print("dof pos lower limits:", props["lower"])
        return props
    
    def _process_rigid_shape_props(self, props, env_id):
        props = super()._process_rigid_shape_props(props, env_id)
        
        # body num != rigid shape num because some bodies have multiple collision shapes, some have none
        if env_id==0:
            self.roller_and_base_shape_indices_to_modify = []
            for i in range(len(self.shape_indices_per_body)):
                print(f"Body {i} name:{self.body_names[i]} has shape indices from: {self.shape_indices_per_body[i].start},  count: {self.shape_indices_per_body[i].count}")
                if self.shape_indices_per_body[i].count != 0 and ("roller_link" in self.body_names[i] or "astribot_torso_base" in self.body_names[i]):
                    # enable collision filter for roller links
                    for j in range(self.shape_indices_per_body[i].start, self.shape_indices_per_body[i].start + self.shape_indices_per_body[i].count):
                        self.roller_and_base_shape_indices_to_modify.append(j)

        return props

    
    
    def compute_observations(self):
        """ Computes observations
            注意对于wheeled base来说, observation里不包含重力投影, 全部关节位置, 和非轮子关节的速度, 因为轮子是速度控制的, 其他关节都是被动的
        """
        full_joint_vel = self.dof_vel.clone()
        wheel_vel = full_joint_vel[:, self.wheel_joint_indices]  # 4 wheels

        # print("===========================print raw obs========================================")
        # print("wheel vel:", wheel_vel)
        # print("base lin vel:", self.base_lin_vel)
        # print("base ang vel:", self.base_ang_vel)
        # print("commands:", self.commands)
        # print("last actions:", self.actions)

        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 3
                                    self.commands[:, :3] * self.commands_scale, # 3
                                    wheel_vel * self.obs_scales.dof_vel, # 4
                                    self.actions # 4
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()
            注意这里输入的action只有四个轮子的速度指令,其他被动关节不在action里,需要对action进行扩充, 被动关节的pd为0,不用担心cimpute_torques那里会出问题

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        full_joint_actions = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        full_joint_actions[:, self.wheel_joint_indices] = self.actions
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(full_joint_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _reward_no_fly(self):
        """
        base lin vel z shoule not exceed threshold, return z vel > threshold
        """
        z_vel_threshold = 5.0  # m/s
        z_vel = self.base_lin_vel[:, 2]
        z_vel_exceed = (z_vel > z_vel_threshold) * 1.0
        return z_vel_exceed
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # print("force:", self.contact_forces[:, self.termination_contact_indices, :])
        # print("force:", self.contact_forces)

        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        # ============================= addeded ==============================
        base_posi_z = self.root_states[:, 2]
        bot_fly = base_posi_z > 0.5
        self.reset_buf |= bot_fly