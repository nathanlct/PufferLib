import numpy as np
import gymnasium
import os
from raylib import rl
import heapq
import math

import pufferlib
from pufferlib.environments.ocean.highway.c_highway import CHighway, step_all
from pufferlib.environments.ocean import render


class PufferHighway(pufferlib.PufferEnv):
    def __init__(self, num_envs=200, render_mode='human'):
        self.num_envs = num_envs
        self.render_mode = render_mode

        # sim hparams (to put in config file)
        self.agents_per_env = 1
        self.cars_per_env = 7
        total_agents = self.num_envs * self.agents_per_env

        self.car_width = 2  # m
        self.car_length = 5  # m
        self.n_lanes = 1

        self.max_speed = 35

        # env spec
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(
            low=-1e9, high=1e9, shape=(1,), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = self.num_envs
        self.render_mode = render_mode
        self.emulated = None
        self.done = False
        self.buf = pufferlib.namespace(
            observations = np.zeros(
                (total_agents, 3), dtype=np.float32),
            rewards = np.zeros(total_agents, dtype=np.float32),
            terminals = np.zeros(total_agents, dtype=bool),
            truncations = np.zeros(total_agents, dtype=bool),
            masks = np.ones(total_agents, dtype=bool),
        )
        self.actions = np.zeros((total_agents, 1), dtype=np.float32)

        self.tick = 0
        self.reward_sum = 0
        self.report_interval = 50

        # env storage
        # veh position is the front bumper position (back bumper is at x = front_bumper - car_length)
        self.veh_positions = np.zeros((self.num_envs, self.cars_per_env), dtype=np.float32)
        self.veh_speeds = np.zeros((self.num_envs, self.cars_per_env), dtype=np.float32)
        self.veh_accels = np.zeros((self.num_envs, self.cars_per_env), dtype=np.float32)
        self.veh_gaps = np.zeros((self.num_envs, self.cars_per_env), dtype=np.float32)
        self.veh_lead_speeds = np.zeros((self.num_envs, self.cars_per_env), dtype=np.float32)
        # (lane numbering increases from leftmost to rightmost lane, starting at 0)
        self.veh_lanes = np.zeros((self.num_envs, self.cars_per_env), dtype=np.uint32)

        # render
        if render_mode == 'human':
            self.client = RaylibClient(
                car_width=self.car_width,
                car_length=self.car_length,
                n_lanes=self.n_lanes,)
    
    def reset(self, seed=None):
        self.c_envs = []
        for i in range(self.num_envs):
            start, end = self.agents_per_env*i, self.agents_per_env*(i+1)
            self.c_envs.append(CHighway(
                self.buf.observations[start:end],
                self.actions[start:end],
                self.buf.rewards[start:end],
                self.veh_positions[i],
                self.veh_speeds[i],
                self.veh_accels[i],
                self.veh_gaps[i],
                self.veh_lead_speeds[i],
                self.car_length,
            ))
            self.c_envs[i].reset()

        return self.buf.observations, {}

    def step(self, actions):
        self.actions[:] = actions
        step_all(self.c_envs)

        info = {}
        self.reward_sum += self.buf.rewards.mean()
        if self.tick % self.report_interval == 0:
            info = {
                'reward': self.reward_sum / self.report_interval,
            }
            self.reward_sum = 0

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, info)

    def render(self):
        if self.render_mode == 'human':
            return self.client.render(self.veh_positions[0], self.veh_speeds[0], self.veh_lanes[0])


class RaylibClient:
    def __init__(self, car_width, car_length, n_lanes):
        self.car_width = car_width
        self.car_length = car_length
        self.n_lanes = n_lanes

        self.screen_width = 1200
        self.screen_height = self.n_lanes * 60
        self.zoom = 1.0

        self.camera_delta = 0

        self.compute_dimensions()

        rl.InitWindow(self.screen_width, self.screen_height, "Highway".encode())
        rl.SetTargetFPS(40)

    def compute_dimensions(self):
        self.lane_height = self.screen_height / self.n_lanes / 2 * self.zoom  # px
        self.road_height = self.lane_height * self.n_lanes
        self.road_y_min = (self.screen_height - self.road_height) / 2
        self.road_y_max = self.road_y_min + self.road_height
        self.lane_edge_dash_length = self.lane_height / 2

        self.ppm = (self.lane_height / self.car_width) / 2  # pixels per meter
        self.lane_veh_padding = (self.lane_height - self.car_width * self.ppm) / 2

    def draw_line(self, x1, y1, x2, y2, color):
        if self.zoom > 0.8:
            rl.DrawLineEx((x1, y1), (x2, y2), 1, color)
        else:
            rl.DrawLine(int(x1), int(y1), int(x2), int(y2), color)

    def render(self, veh_positions, veh_speeds, veh_lanes):
        if rl.IsKeyDown(rl.KEY_ESCAPE):
            exit(0)

        if rl.IsKeyDown(rl.KEY_UP):
            self.zoom += 0.03
            self.zoom = min(self.zoom, 1.8)
            self.compute_dimensions()
        if rl.IsKeyDown(rl.KEY_DOWN):
            self.zoom -= 0.03
            self.zoom = max(self.zoom, 0.2)
            self.compute_dimensions()
        if rl.IsKeyDown(rl.KEY_RIGHT):
            self.camera_delta += 3 / self.zoom
        if rl.IsKeyDown(rl.KEY_LEFT):
            self.camera_delta -= 3 / self.zoom
        if rl.IsKeyDown(rl.KEY_SPACE):
            self.camera_delta = 0

        # tracked position stays at the center of the window
        track_x = veh_positions[0] + self.camera_delta
        min_render_x = track_x * self.ppm - self.screen_width / 2
        max_render_x = min_render_x + self.screen_width

        rl.BeginDrawing()
        rl.ClearBackground(render.PUFF_BACKGROUND)

        # road
        rl.DrawRectangle(0, int(self.road_y_min), self.screen_width, int(self.road_height), [0, 0, 0, 255])
        # solid road edge lines
        self.draw_line(0, self.road_y_min, self.screen_width, self.road_y_min, [255, 255, 255, 255])
        self.draw_line(0, self.road_y_max, self.screen_width, self.road_y_max, [255, 255, 255, 255])
        # dashed lane edge lines
        for i in range(self.n_lanes - 1):
            lane_edge_y = self.road_y_min + (i + 1) * self.lane_height
            lane_edge_x = - (min_render_x % (3 * self.lane_edge_dash_length))
            while lane_edge_x < self.screen_width:
                self.draw_line(lane_edge_x, lane_edge_y, lane_edge_x + self.lane_edge_dash_length, lane_edge_y, [255, 255, 255, 255])
                lane_edge_x += 3 * self.lane_edge_dash_length

        # vehicles 
        for lane, pos in zip(veh_lanes, veh_positions):
            if pos * self.ppm < min_render_x or (pos - self.car_length) * self.ppm > max_render_x:
                continue
            rl.DrawRectangle(
                int(self.screen_width / 2 + (pos - track_x - self.car_length / 2) * self.ppm),
                int(self.road_y_min + lane * self.lane_height + self.lane_veh_padding),
                int(self.car_length * self.ppm),
                int(self.car_width * self.ppm),
                [255, 255, 255, 255]
            )

        rl.EndDrawing()
        return render.cdata_to_numpy()


if __name__ == '__main__':
    env = PufferHighway(num_envs=1, render_mode='human')
    env.reset()
    while True:
        env.step([0] * (env.num_envs * env.agents_per_env))
        env.render()