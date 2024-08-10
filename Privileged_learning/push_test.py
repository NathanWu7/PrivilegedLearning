from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env import (
    ObjectPushEnv,
)
import torch
import os
import numpy as np
import h5py
import imageio
import time
from queue import Queue
from sb3_contrib import TQC
from tactile_gym.Privileged_learning.model2 import Student,mdn_sample

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
    model_dir = './model/push/'
    
    env_modes = {
        # which dofs can have movement (environment dependent)
        # 'movement_mode':'y',
        # 'movement_mode':'yRz',
        #"movement_mode": "xyRz",
        'movement_mode': 'TyRz',
        #'movement_mode':'TxTyRz',

        # specify arm
        "arm_type": "ur5",
        # "arm_type": "mg400",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        # "tactile_sensor_name": "digitac",

        # the type of control used 
        #'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        # randomisations
        "rand_init_orn": True,
        "rand_obj_mass": True,
        "fix_obj":False,

        # straight or random trajectory
        # "traj_type": "straight",
        'traj_type': 'simplex',

        # which observation type to return
        # 'observation_mode':'oracle',
        #"observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',
        # 'observation_mode':'encodedimg_and_feature',
        # 'observation_mode':'encodedimg',
        'observation_mode':'encodedimg_privilege_feature',

        # the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }

    env = ObjectPushEnv(
        max_steps=max_steps,
        env_modes=env_modes,
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=image_size,
    )

    # set seed for deterministic results
    env.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Student(device=device).to(device)
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
            time1 = time.time()
            mu, sigma, pi = model(obs_batch)
            action = mdn_sample(mu, sigma, pi).cpu().numpy()[0]
            time2 = time.time()
            #print(time2-time1)
            #print(action)
            render_img = env.render()
            tactile_img = env.current_img
            render_frames.append(render_img)
            render_tactile.append(tactile_img)
            obs, reward, done, info = env.step(action)
            #print(obs)
            obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
            obs_squence.get() #删除第一张图
            obs_squence.put(obs_data) #插入现在的图

            sum_reward += reward
            if done:
                break
        imageio.mimwrite(os.path.join("videos",str(k)+"render.mp4"), np.stack(render_frames), fps=30)
        imageio.mimwrite(os.path.join("videos",str(k)+"tactile.mp4"), np.stack(render_tactile), fps=30)
        print(sum_reward)
        sum_reward = 0    

 
            
 
    
    

if __name__ == "__main__":
    main()

