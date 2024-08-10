from tactile_gym.rl_envs.exploration.surface_follow.surface_follow_auto.surface_follow_auto_env import (
    SurfaceFollowAutoEnv,
)
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
from sb3_contrib import TQC
import imageio
from tactile_gym.Privileged_learning.model2 import Student, mdn_sample

def main():

    seed = int(0)
    num_iter = 10
    max_steps = 1000
    show_gui = False
    model_test = True
    show_tactile = False
    render = False
    print_info = False
    image_size = [128, 128]
    model_dir = './model/surface/'
    
    env_modes = {
        # which dofs can have movement
        # 'movement_mode':'yz',
        # 'movement_mode':'xyz',
        # 'movement_mode':'yzRx',
        #"movement_mode": "xRz",
        "movement_mode": "xyzRxRy",

        # specify arm
        "arm_type": "ur5",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        # "tactile_sensor_name": "digitac",

        # the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        # noise params for additional robustness
        "noise_mode": "simplex",

        # which observation type to return
        #'observation_mode': 'oracle',
        # "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',
         'observation_mode':'encodedimg_privilege_feature',

        # which reward type to use (currently only dense)
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }

    env = SurfaceFollowAutoEnv(
        max_steps=max_steps,
        env_modes=env_modes,
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=image_size,
    )

    # set seed for deterministic results
    env.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim_dict = env.get_obs_dim()
    input_dim = obs_dim_dict["encodedimg"][0]+obs_dim_dict["extended_feature"][0]
    action_dim = 3
    
    model = Student(input_size = input_dim, action_space=action_dim, device=device).to(device)
    model.load_state_dict(torch.load(model_dir+"_best"))
    obs = env.reset()
    sum_reward = 0
    

    for k in range(10):
        render_frames = []
        render_tactile = []

        obs = env.reset()
        obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
        obs_squence = Queue(maxsize=8) #队列
        for j in range(8):
            obs_squence.put(obs_data)   
        for i in range(400):
            obs_batch = torch.tensor(obs_squence.queue).unsqueeze(0).float().to(device)

            mu, sigma, pi = model(obs_batch)
            action = mdn_sample(mu, sigma, pi).cpu().numpy()[0]

            render_img = env.render()
            tactile_img = env.current_img
            render_frames.append(render_img)
            render_tactile.append(tactile_img)

            obs, reward, done, info = env.step(action)
            
            obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
            obs_squence.get() #删除第一张图
            obs_squence.put(obs_data) #插入现在的图

            sum_reward += reward
            if done:
                break
        print(sum_reward)
        sum_reward = 0    

        imageio.mimwrite(os.path.join("videos2",str(k)+"render.mp4"), np.stack(render_frames), fps=30)
        imageio.mimwrite(os.path.join("videos2",str(k)+"tactile.mp4"), np.stack(render_tactile), fps=30)
            
 
    
    

if __name__ == "__main__":
    main()

