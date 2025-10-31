from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class WheeledBaseFlatCfg( LeggedRobotCfg ):
    """
    Configuration for WheeledBase bot(4 wheels, 10 roller on each wheel) on flat ground.
    """
    class env( LeggedRobotCfg.env):
        num_envs = 2048
        num_observations = 20 # 3(base lin vel, 3 base ang vel, 3 projected_gravity 3 command xy yaw vel, 4 wheel vel, 4 previous actions)
        num_actions = 4
        # episode_length_s = 0.5 # for init pos debug

    # class terrain( LeggedRobotCfg.terrain):
    #     """
    #     Configuration of flat terrain.
    #     """
    #     mesh_type = 'plane' #'trimesh' 'plane'
    #     measure_heights = False
    #     curriculum = False

    class terrain( LeggedRobotCfg.terrain):
        # for train: 减小地形分辨率, 降低collision buffer压力；增加num_cols, terrain_width和border_size, 避免掉落地图边缘
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        horizontal_scale = 1.0 # [m]
        vertical_scale = 0.05 # [m]
        border_size = 100.0
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        max_init_terrain_level = 1 # starting curriculum state
        terrain_length = 20.
        terrain_width = 20.
        num_rows= 2 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.6, 0.4, 0.0, 0.0, 0.0]
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
    
    # class terrain( LeggedRobotCfg.terrain):
    #     # For play
    #     mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
    #     curriculum = True
    #     horizontal_scale = 0.1 # [m]
    #     vertical_scale = 0.005 # [m]
    #     border_size = 10.0
    #     static_friction = 1.0
    #     dynamic_friction = 1.0
    #     restitution = 0.
    #     max_init_terrain_level = 1 # starting curriculum state
    #     terrain_length = 8.
    #     terrain_width = 8.
    #     num_rows= 2 # number of terrain rows (levels)
    #     num_cols = 3 # number of terrain cols (types)
    #     # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    #     # terrain_proportions = [0.6, 0.4, 0.0, 0.0, 0.0]
    #     terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    #     # trimesh only:
    #     slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
    
    class commands( LeggedRobotCfg.commands ):
        heading_command = True
        class ranges( LeggedRobotCfg.commands.ranges ):
            # lin_vel_x = [-1.0, 1.0] # min max [m/s]
            # lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # lin_vel_x = [0.5, 0.5] # min max [m/s]
            # lin_vel_y = [0.5, 0.5]   # min max [m/s]
            lin_vel_x = [1.0, 1.0] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            heading = [0.0, 0.0]


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.11]
        # rot = [0, 0, 0.3826834, 0.9238795] # yaw 45度
        default_joint_angles = {
            # Wheels
            'wheel_RF_Joint': 0.0,
            'wheel_LF_Joint': 0.0,
            'wheel_RR_Joint': 0.0,
            'wheel_LR_Joint': 0.0,

            # Rollers on wheels total 80, each wheel 20
            'wheel_RF_roller_Joint_1_1': 0.0,
            'wheel_RF_roller_Joint_1_2': 0.0,
            'wheel_RF_roller_Joint_2_1': 0.0,
            'wheel_RF_roller_Joint_2_2': 0.0,
            'wheel_RF_roller_Joint_3_1': 0.0,
            'wheel_RF_roller_Joint_3_2': 0.0,
            'wheel_RF_roller_Joint_4_1': 0.0,
            'wheel_RF_roller_Joint_4_2': 0.0,
            'wheel_RF_roller_Joint_5_1': 0.0,
            'wheel_RF_roller_Joint_5_2': 0.0,
            'wheel_RF_roller_Joint_6_1': 0.0,
            'wheel_RF_roller_Joint_6_2': 0.0,
            'wheel_RF_roller_Joint_7_1': 0.0,
            'wheel_RF_roller_Joint_7_2': 0.0,
            'wheel_RF_roller_Joint_8_1': 0.0,
            'wheel_RF_roller_Joint_8_2': 0.0,
            'wheel_RF_roller_Joint_9_1': 0.0,
            'wheel_RF_roller_Joint_9_2': 0.0,
            'wheel_RF_roller_Joint_10_1': 0.0,
            'wheel_RF_roller_Joint_10_2': 0.0,
            # similarly for other three wheels' rollers...
            'wheel_LF_roller_Joint_1_1': 0.0,
            'wheel_LF_roller_Joint_1_2': 0.0,
            'wheel_LF_roller_Joint_2_1': 0.0,
            'wheel_LF_roller_Joint_2_2': 0.0,
            'wheel_LF_roller_Joint_3_1': 0.0,
            'wheel_LF_roller_Joint_3_2': 0.0,
            'wheel_LF_roller_Joint_4_1': 0.0,
            'wheel_LF_roller_Joint_4_2': 0.0,
            'wheel_LF_roller_Joint_5_1': 0.0,
            'wheel_LF_roller_Joint_5_2': 0.0,
            'wheel_LF_roller_Joint_6_1': 0.0,
            'wheel_LF_roller_Joint_6_2': 0.0,
            'wheel_LF_roller_Joint_7_1': 0.0,
            'wheel_LF_roller_Joint_7_2': 0.0,
            'wheel_LF_roller_Joint_8_1': 0.0,
            'wheel_LF_roller_Joint_8_2': 0.0,
            'wheel_LF_roller_Joint_9_1': 0.0,
            'wheel_LF_roller_Joint_9_2': 0.0,
            'wheel_LF_roller_Joint_10_1': 0.0,
            'wheel_LF_roller_Joint_10_2': 0.0,
            'wheel_RR_roller_Joint_1_1': 0.0,
            'wheel_RR_roller_Joint_1_2': 0.0,
            'wheel_RR_roller_Joint_2_1': 0.0,
            'wheel_RR_roller_Joint_2_2': 0.0,
            'wheel_RR_roller_Joint_3_1': 0.0,
            'wheel_RR_roller_Joint_3_2': 0.0,
            'wheel_RR_roller_Joint_4_1': 0.0,
            'wheel_RR_roller_Joint_4_2': 0.0,
            'wheel_RR_roller_Joint_5_1': 0.0,
            'wheel_RR_roller_Joint_5_2': 0.0,
            'wheel_RR_roller_Joint_6_1': 0.0,
            'wheel_RR_roller_Joint_6_2': 0.0,
            'wheel_RR_roller_Joint_7_1': 0.0,
            'wheel_RR_roller_Joint_7_2': 0.0,
            'wheel_RR_roller_Joint_8_1': 0.0,
            'wheel_RR_roller_Joint_8_2': 0.0,
            'wheel_RR_roller_Joint_9_1': 0.0,
            'wheel_RR_roller_Joint_9_2': 0.0,
            'wheel_RR_roller_Joint_10_1': 0.0,
            'wheel_RR_roller_Joint_10_2': 0.0,
            'wheel_LR_roller_Joint_1_1': 0.0,
            'wheel_LR_roller_Joint_1_2': 0.0,
            'wheel_LR_roller_Joint_2_1': 0.0,
            'wheel_LR_roller_Joint_2_2': 0.0,
            'wheel_LR_roller_Joint_3_1': 0.0,
            'wheel_LR_roller_Joint_3_2': 0.0,
            'wheel_LR_roller_Joint_4_1': 0.0,
            'wheel_LR_roller_Joint_4_2': 0.0,
            'wheel_LR_roller_Joint_5_1': 0.0,
            'wheel_LR_roller_Joint_5_2': 0.0,
            'wheel_LR_roller_Joint_6_1': 0.0,
            'wheel_LR_roller_Joint_6_2': 0.0,
            'wheel_LR_roller_Joint_7_1': 0.0,
            'wheel_LR_roller_Joint_7_2': 0.0,
            'wheel_LR_roller_Joint_8_1': 0.0,
            'wheel_LR_roller_Joint_8_2': 0.0,
            'wheel_LR_roller_Joint_9_1': 0.0,
            'wheel_LR_roller_Joint_9_2': 0.0,
            'wheel_LR_roller_Joint_10_1': 0.0,
            'wheel_LR_roller_Joint_10_2': 0.0,
        }
    
    class control( LeggedRobotCfg.control ):
        # 轮子采用速度控制
        # TODO: 不同joint可以采用不同的控制模式，比如轮子速度控制，身体采用位置控制, 或全部力距控制
        control_type = 'V'

        # PD Drive parameters:
        # 大的p有助于前向20的关节跟踪和爬坡, 也有助于rl训练cartsian 1m/s爬坡, 但容易导致翘头, 加100的力距保护能缓解翘头terminate
        stiffness = {
            'wheel_RF_Joint': 2.0,
            'wheel_LF_Joint': 2.0,
            'wheel_RR_Joint': 2.0,
            'wheel_LR_Joint': 2.0,

            'roller_Joint': 0.0,
        }
        damping = {
            'wheel_RF_Joint': 0.0001,
            'wheel_LF_Joint': 0.0001,
            'wheel_RR_Joint': 0.0001,
            'wheel_LR_Joint': 0.0001,

            'roller_Joint': 0.0,
        }
        # action scale: target velocity = actionScale * action
        action_scale = 10.0
        decimation = 10

    class asset( LeggedRobotCfg.asset ):
        file = "/home/zxw/legged_gym/resources/robots/s1_base/urdf/astribot_s1_base.urdf"
        name = "astribot_s1_base"
        foot_name = "roller_link"
        penalize_contacts_on = ['astribot_torso_base']
        terminate_after_contacts_on = ['astribot_torso_base']
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        self_collisions = 0 # 0 to enable self-collisions(disable filters), 1 to disable self-collisions(enable filters)
        armature = 0.0

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 1000.
        only_positive_rewards = False

        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.6
            torques = -2.e-7
            dof_acc = -1.e-9
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            feet_air_time = 0.0
            dof_pos_limits = -0.0
            feet_contact_forces = -0.
            no_fly = -10.0
            # upturned_head_at_begining = -0.1
            angle_vel_xy_at_begining = -0.1
    
    class sim:
        dt =  0.002
        substeps = 2
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class WheeledBaseFlatCfgPPO( LeggedRobotCfgPPO ):

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_wheeled_base'

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    