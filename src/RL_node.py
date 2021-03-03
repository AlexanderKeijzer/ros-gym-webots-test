#!/usr/bin/env python3

import gym
import numpy as np
import time
import types
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from gym_env.ros_env import ROSEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


if __name__ == '__main__':
    rospy.init_node('gym_env',
                    anonymous=True, log_level=rospy.WARN)

    env = ROSEnv(['/joint_sensors'], 
                 ['/joint_actuators'],
                  '/reward',
                  '/webots')

    #rospy.sleep(15)

    check_env(env)

    rospy.loginfo("Training starts")

    model = PPO('MlpPolicy', env, verbose=1)
    
    model.learn(total_timesteps=10000)

    obs = env.reset()
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    
    env.close()