import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.spaces import Box
from gymnasium.utils import seeding
import numpy as np
import time
import vtk
import threading
import itertools
import copy

from gym_uav.envs.utils import TimerCallback
from gym_uav.envs.utils import Config
from gym_uav.envs.utils import Smoother, Smoother_soft
from gym_uav.envs.utils import add_arguments
import argparse


class UavDenseEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        Common = Config()
        self.Common = Common
        self.basic_directions = Common.basic_directions
        self.extra_directions = Common.extra_directions
        self.original_observation_length = Common.original_observation_length
        self.extra_length = len(self.extra_directions)

        self.observation_space = Box(-np.inf, np.inf, [self.original_observation_length + self.extra_length], float)
        self.action_space = Box(-1.0, 1.0, [2], float)
        self._env_step_counter = 0
        self.state = np.zeros([self.observation_space.shape[0]])  

        # uav parameters
        self.level = Common.level
        self.position = np.zeros([2])
        self.target = np.zeros([2])
        self.orient = np.zeros([1])
        self.speed = np.zeros([1])  
        self.max_speed = Common.max_speed
        self.min_distance_to_target = Common.min_distance_to_target
        self.real_action_range = Common.real_action_range

        # environment parameters
        self.min_distance_to_obstacle = Common.min_distance_to_obstacle
        self.min_initial_starts = Common.min_initial_starts
        self.expand = Common.expand
        self.num_circle = Common.num_circle
        self.radius = Common.radius
        self.period = Common.period
        self.mat_height = None 
        self.mat_exist = None 
        self.lowest = Common.lowest
        self.delta = Common.delta
        self.total = Common.total

        # reward parameters
        self.use_sparse_reward = Common.use_sparse_reward
        if self.use_sparse_reward:
            self.obstacle_coef = Common.obstacle_coef_sparse
            self.action_coef = Common.action_coef_sparse
            self.step_coef = Common.step_coef_sparse
            self.distance_coef = Common.distance_coef_sparse
            self.goal_coef = Common.goal_coef_sparse
            self.crash_coef = Common.crash_coef_sparse
        else:
            self.obstacle_coef = Common.obstacle_coef
            self.action_coef = Common.action_coef
            self.step_coef = Common.step_coef
            self.distance_coef = Common.distance_coef
            self.goal_coef = Common.goal_coef
            self.crash_coef = Common.crash_coef

        # curriculum parameters
        self.use_curriculum_learning = Common.use_curriculum_learning
        self.curriculum_height = Common.curriculum_height
        self.curriculum_reach_goal_distance = Common.curriculum_reach_goal_distance
        self.curriculum_min_init_distance = Common.curriculum_min_init_distance
        self.curriculum_map_size = Common.curriculum_map_size

        # range finder parameters
        self.scope = Common.scope
        self.min_step = Common.min_step
        self.directions = self.basic_directions + self.extra_directions
        self.end_points = [None for _ in range(len(self.directions))]

        # rendering parameters
        self.margin = Common.margin
        self.env_params = {'cylinders': None, 'size': 1.5*(self.num_circle+self.margin*2)*self.period,
                           'departure': None, 'arrival': None}
        self.agent_params = {'position': self.position, 'target': self.target, 'direction':None, 'rangefinders': self.end_points}
        self.agent_params_pre = None
        self.first_render = True
        self.terminate_render = False
        self.camera_alpha = Common.camera_alpha

        # other parameters
        self.is_reset = False

        assert self.scope > self.max_speed

    def _fast_range_finder(self, position, theta, forward_dist, min_dist=0.0, find_type='normal'):
        end_cache = copy.deepcopy(position)
        position_integer = np.floor(end_cache / self.period).astype(np.int32)
        judge = end_cache - (position_integer * self.period + self.period / 2)
        if judge[0] >= 0 and judge[1] > 0:
            down_left = position_integer * self.period + self.period / 2
            down_right = (position_integer + np.array([1, 0])) * self.period + self.period / 2
            up_left = (position_integer + np.array([0, 1])) * self.period + self.period / 2
            up_right = (position_integer + np.array([1, 1])) * self.period + self.period / 2
            exists = dict(zip(['down_left', 'down_right', 'up_left', 'up_right'],
                              [self.mat_exist[position_integer[0] + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[position_integer[0] + 1 + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[position_integer[0] + self.expand, position_integer[1] + 1 + self.expand],
                               self.mat_exist[
                                   position_integer[0] + 1 + self.expand, position_integer[1] + 1 + self.expand]]))
        elif judge[0] >= 0 and judge[1] < 0:
            down_left = (position_integer + np.array([0, -1])) * self.period + self.period / 2
            down_right = (position_integer + np.array([1, -1])) * self.period + self.period / 2
            up_left = position_integer * self.period + self.period / 2
            up_right = (position_integer + np.array([1, 0])) * self.period + self.period / 2
            exists = dict(zip(['down_left', 'down_right', 'up_left', 'up_right'],
                              [self.mat_exist[position_integer[0] + self.expand, position_integer[1] - 1 + self.expand],
                               self.mat_exist[position_integer[0] + 1 + self.expand, position_integer[1] - 1 + self.expand],
                               self.mat_exist[position_integer[0] + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[
                                   position_integer[0] + 1 + self.expand, position_integer[1] + self.expand]]))
        elif judge[0] < 0 and judge[1] > 0:
            down_left = (position_integer + np.array([-1, 0])) * self.period + self.period / 2
            down_right = position_integer * self.period + self.period / 2
            up_left = (position_integer + np.array([-1, 1])) * self.period + self.period / 2
            up_right = (position_integer + np.array([0, 1])) * self.period + self.period / 2
            exists = dict(zip(['down_left', 'down_right', 'up_left', 'up_right'],
                              [self.mat_exist[position_integer[0] -1 + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[
                                   position_integer[0] + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[position_integer[0] - 1 + self.expand, position_integer[1] + 1 + self.expand],
                               self.mat_exist[
                                   position_integer[0] + self.expand, position_integer[1] + 1 + self.expand]]))
        else:
            down_left = (position_integer + np.array([-1, -1])) * self.period + self.period / 2
            down_right = (position_integer + np.array([0, -1])) * self.period + self.period / 2
            up_left = (position_integer + np.array([-1, 0])) * self.period + self.period / 2
            up_right = position_integer * self.period + self.period / 2
            exists = dict(zip(['down_left', 'down_right', 'up_left', 'up_right'],
                              [self.mat_exist[position_integer[0] - 1 + self.expand, position_integer[1] - 1 + self.expand],
                               self.mat_exist[
                                   position_integer[0] + self.expand, position_integer[1] - 1 + self.expand],
                               self.mat_exist[
                                   position_integer[0] - 1 + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[
                                   position_integer[0] + self.expand, position_integer[1] + self.expand]]))

        base_points = dict(zip(['down_left', 'down_right', 'up_left', 'up_right'], [down_left, down_right, up_left, up_right]))

        dist = []
        end = []
        for base in base_points.keys():
            theta_base = np.arctan(np.abs((base_points[base] - end_cache)[0] / (base_points[base] - end_cache)[1]))
            if base == 'down_left':
                theta_base = np.pi + theta_base
            if base == 'down_right':
                theta_base = np.pi - theta_base
            if base == 'up_left':
                theta_base = 2 * np.pi - theta_base
            if base == 'up_right':
                theta_base = theta_base

            theta_base = np.mod(theta_base, 2*np.pi)

            dist_to_base = np.linalg.norm(end_cache - base_points[base])

            delta_theta = theta - theta_base
            if dist_to_base - (self.radius + min_dist) >= forward_dist or exists[base] < 0:
                dist.append(1.0)
                end.append(end_cache + np.array([forward_dist * np.sin(theta[0]), forward_dist * np.cos(theta[0])]))
            elif (dist_to_base - (self.radius + min_dist) >= 0) and (np.cos(delta_theta) <= 0):
                dist.append(1.0)
                end.append(end_cache + np.array([forward_dist * np.sin(theta[0]), forward_dist * np.cos(theta[0])]))
            else:
                min_dist_to_origin = np.abs(np.sin(delta_theta)) * dist_to_base

                if min_dist_to_origin >= (self.radius + min_dist):
                    dist.append(1.0)
                    end.append(end_cache + np.array([forward_dist * np.sin(theta[0]), forward_dist * np.cos(theta[0])]))
                else:
                    dist_inner = np.sqrt((self.radius + min_dist) ** 2 - min_dist_to_origin ** 2)
                    final_dist = np.cos(delta_theta) * dist_to_base - dist_inner
                    final_dist = final_dist[0]
                    if final_dist >= forward_dist:
                        dist.append(1.0)
                        end.append(end_cache + np.array([forward_dist * np.sin(theta[0]), forward_dist * np.cos(theta[0])]))
                    else:
                        dist.append(final_dist / forward_dist)
                        end.append(end_cache + np.array([final_dist * np.sin(theta[0]), final_dist * np.cos(theta[0])]))
        dist = np.array(dist)

        return np.min(dist), end[np.argmin(dist)]

    def _range_finder(self, position, theta, steps, min_dist=0.0, find_type='normal'):
        end_cache = copy.deepcopy(position)
        Count = 0
        state = 1.0

        while Count < steps:
            Count = Count + 1
            end_cache = end_cache + np.array([self.min_step * np.sin(theta[0]), self.min_step * np.cos(theta[0])])
            end = np.mod(end_cache, self.period)

            position_integer = np.floor(end_cache / self.period).astype(np.int32)
            if self.mat_exist[position_integer[0] + self.expand, position_integer[1] + self.expand] > 0:
                if np.linalg.norm(end - np.array([self.period / 2, self.period / 2])) - (self.radius + min_dist) <= 0:
                    state = np.linalg.norm(end_cache - position) / self.scope
                    break

        return state, end_cache

    def _prepare_background_for_render(self):
        small_mat_height = self.mat_height[self.expand - self.margin: self.expand + self.num_circle + self.margin,
                           self.expand - self.margin: self.expand + self.num_circle + self.margin]
        small_mat_exist = self.mat_exist[self.expand - self.margin: self.expand + self.num_circle + self.margin,
                          self.expand - self.margin: self.expand + self.num_circle + self.margin]
        index_tmp = [i - self.margin for i in range(np.shape(small_mat_height)[0])]
        position_tmp = list(itertools.product(index_tmp, index_tmp))
        position_tmp = [list(pos) for pos in position_tmp]
        position_tmp = np.array(position_tmp)
        position_tmp = position_tmp * self.period + self.period / 2

        cylinders = []
        small_mat_height = list(small_mat_height.reshape(1,-1)[0])
        small_mat_exist = list(small_mat_exist.reshape(1,-1)[0])
        position_tmp = list(position_tmp)

        for hei, exi, pos in zip(small_mat_height, small_mat_exist, position_tmp):
            if exi > 0:
                p1 = np.concatenate([pos, np.array([0])])
                p2 = np.concatenate([pos, np.array([hei])])
                r = self.radius
                cylinders.append([p1, p2, r])

        flight_height = self.level if not self.use_curriculum_learning else int(self.curriculum_height)
        self.env_params['cylinders'] = copy.deepcopy(cylinders)
        self.env_params['departure'] = copy.deepcopy(np.concatenate([self.position, np.array([flight_height])]))
        self.env_params['arrival'] = copy.deepcopy(np.concatenate([self.target, np.array([flight_height])]))

    def _get_observation(self, position, target, orient):
        global_counter = 0
        basic_counter = 0
        extra_counter = 0
        flight_height = self.level if not self.use_curriculum_learning else int(self.curriculum_height)

        for dir in self.basic_directions:
            theta = np.mod(dir + orient, 2 * np.pi)  
            self.state[basic_counter], end_cache = self._fast_range_finder(position, theta, self.scope)
            self.end_points[global_counter] = [np.concatenate([position, np.array([flight_height])]),
                                                   np.concatenate([end_cache, np.array([flight_height])])]

            global_counter += 1
            basic_counter += 1

        # adding extra range finders
        for dir in self.extra_directions:
            theta = np.mod(dir + orient, 2 * np.pi)  
            self.state[15 + extra_counter], end_cache = self._fast_range_finder(position, theta, self.scope)
            self.end_points[global_counter] = [np.concatenate([position, np.array([flight_height])]),
                                                   np.concatenate([end_cache, np.array([flight_height])])]
            global_counter += 1
            extra_counter += 1

        dist = np.linalg.norm(target - position)
        self.state[9] = 2*(dist / (np.sqrt(2)*self.period * self.num_circle) - 0.5)
        theta_target = np.arctan((target[0] - position[0]) / (target[1] - position[1]))
        if (target[0] >= position[0]) and (target[1] >= position[1]):
            self.state[10] = np.sin(theta_target)
            self.state[11] = np.cos(theta_target)
        elif target[1] < position[1]:
            self.state[10] = np.sin(theta_target + np.pi)
            self.state[11] = np.cos(theta_target + np.pi)
        else:
            self.state[10] = np.sin(theta_target + 2 * np.pi)
            self.state[11] = np.cos(theta_target + 2 * np.pi)

        self.state[12] = np.sin(orient)  # normalization
        self.state[13] = np.cos(orient)  # normalization
        self.state[14] = 2*(self.speed / self.max_speed - 0.5)

        flight_height = self.level if not self.use_curriculum_learning else int(self.curriculum_height)
        self.agent_params_pre = copy.deepcopy(self.agent_params)
        self.agent_params['position'] = copy.deepcopy(np.concatenate([position, np.array([flight_height])]))
        self.agent_params['target'] = copy.deepcopy(np.concatenate([target, np.array([flight_height])]))
        self.agent_params['rangefinders'] = copy.deepcopy(self.end_points)
        self.agent_params['direction'] = copy.deepcopy(np.mod(90 - orient/2/np.pi*360, 360))
        self.agent_params['direction_camera'] = copy.deepcopy(np.mod(90 - np.mod(self.orient_render, 2*np.pi)/2/np.pi*360, 360))

    def _get_reward(self, position_prev, position_curr, target, range_finders, action, info, done1, done2):
        reward_sparse = self.goal_coef if done2 else 0.0
        reward_distance = self.distance_coef * (np.linalg.norm(position_prev - target) - np.linalg.norm(position_curr - target))
        reward_barrier = -self.obstacle_coef  if np.min(range_finders) * self.scope <= 10.0 else 0.0
        reward_step = -self.step_coef
        reward_action = -self.action_coef * np.linalg.norm(action)
        reward_crash = -self.crash_coef if done1 else 0.0

        reward = reward_sparse + reward_barrier + reward_distance + reward_step + reward_action + reward_crash
        info['goal_rew'] = reward_sparse
        info['obstacle_pen'] = reward_barrier
        info['distance_rew'] = reward_distance
        info['step_pen'] = reward_step
        info['action_pen'] = reward_action
        info['crash_pen'] = reward_crash
        info['total_rew'] = reward
        return reward 

    def step(self, action):
        assert self.is_reset, 'the environment must be reset before it is called'
        self._env_step_counter += 1
        position_temp = copy.deepcopy(self.position)
        self.orient = np.mod(self.real_action_range[0] * action[0] * np.pi + self.orient, 2 * np.pi)

        self.orient_total_pre = copy.deepcopy(self.orient_total)
        self.orient_render_pre = copy.deepcopy(self.orient_render)

        self.orient_total = self.real_action_range[0] * action[0] * np.pi + self.orient_total
        self.orient_render = self.orient_total * self.camera_alpha\
                             + self.orient_render * (1-self.camera_alpha)

        self.speed = np.where(action[1] >= 0,
                              self.speed + self.real_action_range[1] * action[1] * (-np.tanh(0.5 * (self.speed - self.max_speed))),
                              self.speed + self.real_action_range[1] * action[1] * np.tanh(0.5 * self.speed))

        iter_num = np.where(self.speed / self.min_step > np.floor(self.speed / self.min_step),
                            np.int32(self.speed / self.min_step) + 1, np.int32(self.speed / self.min_step))[0]
        done1, end_cache = self._fast_range_finder(np.copy(self.position), np.copy(self.orient), self.speed[0],
                                                   self.min_distance_to_obstacle, 'forward')

        self.position = np.copy(end_cache)

        self._get_observation(np.copy(self.position), np.copy(self.target), np.copy(self.orient))
        next_observation = np.copy(self.state)

        done1 = True if done1 < 1.0 else False
        if self.use_curriculum_learning:
            done2 = (np.linalg.norm(self.position - self.target) <= self.curriculum_reach_goal_distance) 
        else:
            done2 = (np.linalg.norm(self.position - self.target) <= self.min_distance_to_target)
        done3 = (np.linalg.norm(self.position - self.target) >= 1e4)

        done = done1 + done2 + done3

        if self._env_step_counter == 100:
            if done:
                truncated = False 
            else:
                truncated = True
        else:
            truncated = False
            
        info = {}
        reward = self._get_reward(position_temp, self.position, self.target, self.state[0:9], action, info, done1, done2)
        if done:
            if done2:
                info.update({'is_success': True})
            else:
                info.update({'is_success': False})
            if done1:
                info.update({'is_crash': True})
            if done3:
                info.update({'is_termination': True})
            self.terminate_render = True
        else:
            if truncated:
                info.update({'TimeLimit.truncated': True})
                info.update({'terminal_observation': copy.deepcopy(next_observation)})
                info.update({'is_success': False})

        return next_observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        if isinstance(options, argparse.Namespace):
            self.use_curriculum_learning = options.use_curriculum_learning
            self.use_sparse_reward = options.use_sparse_reward
            print("=============== reset environment mode=============== ")
            print("use_sparse_reward={}".format(self.use_sparse_reward))
            print("use_curriculum_learning={}".format(self.use_curriculum_learning))
            print("====================================================")
            if self.use_sparse_reward:
                self.obstacle_coef = self.Common.obstacle_coef_sparse
                self.action_coef =  self.Common.action_coef_sparse
                self.step_coef =  self.Common.step_coef_sparse
                self.distance_coef =  self.Common.distance_coef_sparse
                self.goal_coef =  self.Common.goal_coef_sparse
                self.crash_coef =  self.Common.crash_coef_sparse
            else:
                self.obstacle_coef =  self.Common.obstacle_coef
                self.action_coef =  self.Common.action_coef
                self.step_coef =  self.Common.step_coef
                self.distance_coef =  self.Common.distance_coef
                self.goal_coef =  self.Common.goal_coef
                self.crash_coef =  self.Common.crash_coef
        if options is not None and not isinstance(options, argparse.Namespace) and self.use_curriculum_learning and 'update_curriculum_params' in options and options['update_curriculum_params']:
            self.curriculum_reach_goal_distance = max(self.min_distance_to_target, self.curriculum_reach_goal_distance / 1.2000)
            self.curriculum_min_init_distance = min(self.min_initial_starts, self.curriculum_min_init_distance * 1.2000)
            self.curriculum_map_size = min(self.num_circle, self.curriculum_map_size *1.2000)
            self.curriculum_height = max(self.level, self.curriculum_height / 1.2000)
            print("=============== curriculum update ==========================")
            print("curriculum_reach_goal_distance is:", self.curriculum_reach_goal_distance)
            print("curriculum_min_init_distance is:", self.curriculum_min_init_distance)
            print("curriculum_map_size is:", self.curriculum_map_size)
            print("curriculum_height is:", self.curriculum_height)
            print("==========================================================")
            reset_info= {}
            reset_info['curriculum_reach_goal_distance'] = self.curriculum_reach_goal_distance
            reset_info['curriculum_min_init_distance'] = self.curriculum_min_init_distance
            reset_info['curriculum_map_size'] = self.curriculum_map_size
            reset_info['curriculum_height'] = self.curriculum_height
        else:
            reset_info = {}

        if seed is not None:  
            np.random.seed = seed 

        self.is_reset = True
        self.first_render = True
        self.terminate_render = False
        self._env_step_counter = 0
        self.mat_height = np.random.randint(
            1, self.total, size=(self.num_circle + 2 * self.expand, self.num_circle + 2 * self.expand)) * self.delta + self.lowest  
        W, H = np.shape(self.mat_height)

        flight_height = self.level if not self.use_curriculum_learning else int(self.curriculum_height)
        for i in range(W):
            for j in range(H):
                if self.mat_height[i, j] > flight_height + self.delta:
                    self.mat_height[i, j] = self.mat_height[i, j] + np.int32(np.random.uniform(0, 200))

        self.mat_exist = self.mat_height - flight_height  
        
        while True:
            position = np.random.uniform(0, self.period, size=(2,))
            if np.linalg.norm(position - np.array([self.period / 2, self.period / 2])) - (self.radius + self.min_distance_to_obstacle) > 0:
                break
        relative_position = np.random.randint(0, self.num_circle, size=(2,)).astype(np.float64) if not self.use_curriculum_learning else np.random.randint(0, int(self.curriculum_map_size), size=(2,)).astype(np.float64)
        self.position = position + relative_position * self.period

        while True:
            target = np.random.uniform(0, self.period, size=(2,))
            if np.linalg.norm(target - np.array([self.period / 2, self.period / 2])) - self.radius > 0:
                break

        counter = 0
        init_distance = self.min_initial_starts if not self.use_curriculum_learning else self.curriculum_min_init_distance
        while True:  # ensure that the minimum distance between initial position and target position is larger than 200m
            counter += 1
            relative_target = np.random.randint(0, self.num_circle, size=(2,)).astype(np.float64) if not self.use_curriculum_learning else np.random.randint(0, int(self.curriculum_map_size), size=(2,)).astype(np.float64)
            target_temp = np.array(target + relative_target * self.period)
            if np.linalg.norm(target_temp - self.position) >= init_distance:
                self.target = target + relative_target * self.period
                self.orient = np.random.uniform(0, 2 * np.pi, size=(1,))
                self.orient_render = copy.deepcopy(self.orient)
                self.orient_total = copy.deepcopy(self.orient)
                self.orient_render_pre = copy.deepcopy(self.orient_render)
                self.orient_total_pre = copy.deepcopy(self.orient_total)
                self.speed = np.zeros([1])
                self._prepare_background_for_render()
                self._get_observation(np.copy(self.position), np.copy(self.target), np.copy(self.orient))
                observation = np.copy(self.state)
                break
            elif counter > 20:
                print("reset again")
                return self.reset()
            else:
                pass
        return observation, reset_info

    def render(self, mode='human'):
        sleep_time = 0.1
        print('orient={}, speed={}'.format(self.orient, self.speed), self.orient_render)

        assert self.is_reset, 'the environment must be reset before rendering'
        if self.first_render:
            time.sleep(sleep_time)
            self.first_render = False
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(.2, .2, .2)
            # Render Window
            renderWindow = vtk.vtkRenderWindow()
            renderWindow.AddRenderer(renderer)
            renderWindow.SetSize(1600, 1600)
            self.Timer = TimerCallback(renderer)
            self.Timer.env_params = self.env_params

            def environment_render():
                renderWindowInteractor = vtk.vtkRenderWindowInteractor()
                renderWindowInteractor.SetRenderWindow(renderWindow)
                renderWindowInteractor.Initialize()
                renderWindowInteractor.AddObserver('TimerEvent', self.Timer.execute)
                timerId = renderWindowInteractor.CreateRepeatingTimer(30)
                self.Timer.timerId = timerId

                renderWindow.Start()
                renderWindowInteractor.Start()

            self.th = threading.Thread(target=environment_render, args=())
            self.th.start()
            time.sleep(1.0)
        else:
            self.Timer.terminate_render = self.terminate_render

            positions, directions, directions_camera = \
                Smoother_soft(self.agent_params_pre['position'], self.agent_params['position'],
                              self.orient_total_pre[0],
                              self.orient_total[0],
                              self.orient_render_pre[0],
                              self.orient_render[0])

            for i in range(len(directions)):
                directions[i] = np.array([np.mod(90 - np.mod(directions[i] / 2 / np.pi * 360.0, 360), 360)])

            for i in range(len(directions_camera)):
                directions_camera[i] = np.array([np.mod(165 - np.mod(directions_camera[i] / 2 / np.pi * 360.0, 360), 360)])

            for i in range(len(positions)):
                time.sleep(sleep_time / len(positions))
                agent_params_tmp = copy.deepcopy(self.agent_params)
                agent_params_tmp['position'] = positions[i]
                agent_params_tmp['direction'] = directions[i]
                agent_params_tmp['direction_camera'] = directions_camera[i]
                if i < len(positions) - 1:
                    agent_params_tmp['rangefinders'] = None
                self.Timer.agent_params = agent_params_tmp
        return self.speed, self.orient

    def seed(self, seed=None):
        if seed:
            if seed >= 0:
                np.random.seed(seed)
    #
    # def close(self):
    #     pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    env = UavDenseEnv(parser)
    env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        action[0] = 0.25 * action[0]
        obe, rew, done, info = env.step(action)
        print("reward", rew)
        env.render()
        if done:
            exit(0)


