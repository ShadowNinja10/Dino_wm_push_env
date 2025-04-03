"""This example showcase an arrow pointing or aiming towards the cursor.
"""

__docformat__ = "reStructuredText"

import sys
import os
import pygame
import pickle
from copy import deepcopy

import pymunk
import pymunk.constraints
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
from pymunk.autogeometry import march_soft, march_hard
from IPython import embed
import cv2  
import torch  
from copy import deepcopy


import gym
import einops
from gym import spaces
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
from typing import Tuple, Sequence, Dict, Union, Optional
import pygame
import pymunk
import numpy as np
import shapely.geometry as sg
import cv2
import skimage.transform as st
import pymunk.pygame_util
import collections
from matplotlib import cm
import torch


# Set the dummy video driver before reinitializing pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'

square_pts = [(40, 40), (40, -40), (-40, -40), (-40, 40)]
trapezoid_pts = [(20, 20), (40, -40), (-40, -40), (-20, 20)]
T_pts = [(40, 40), (40, 20), (10, 20), (10, -40), (-10, -40), (-10, 20), (-40, 20), (-40, 40)]

L_img = [
                "        ",
                "  xx    ",
                "  xx    ",
                "  xx    ",
                "  xx    ",
                "  xx    ",
                "  xxxxx ",
                "  xxxxx ",
                "        "
            ]

T_img = [
                "         ",
                "  xxxxxx ",
                "  xxxxxx ",
                "    xx   ",
                "    xx   ",
                "    xx   ",
                "    xx   ",
                "    xx   ",
                "         "
            ]

E_img = [
                "        ",
                " xxxxxx ",
                " xxxxxx ",
                " xx     ",
                " xx     ",
                " xxxxxx ",
                " xxxxxx ",
                " xx     ",
                " xx     ",
                " xxxxxx ",
                " xxxxxx ",
                "        "
            ]

X_img = [
                "           ",
                " xxx   xxx ",
                "  xxx xxx  ",
                "   xxxxx   ",
                "    xxx    ",
                "   xxxxx   ",
                "  xxx xxx  ",
                " xxx   xxx ",
                "           "
                            ]


block_img_dict={
    "X_img":X_img,
    "E_img":E_img,
    "T_img":T_img,
    "L_img":L_img
}



OPTIONS = {
    'screen_size': (600, 600),
    'damping': 0.01,
    'dt': 1.0/60.0,
    'boundary_pts': [(10, 10), (590, 10), (590, 590), (10, 590)],
    'block_pts': T_pts,
    'block_img': X_img,
    'block_img_scale': 10,
    'block_img_flag': True, 
    'block_mass': 10,
    'block_start_pos': (300, 300),
    'pusher_start_pos': (300, 220),
    'target_start_pos': (200, 400),
    'target_start_angle': np.pi/6,
    'pusher_mass': 10,
    'pusher_radius': 10,
    'elasticity': 0.0,
    'friction': 1.0,
    'block_color': (254, 33, 139, 255),
    'pusher_color': (33, 176, 254, 255.),
    'target_color': (254, 215, 0, 0.),
    'controller_stiffness': 10000,
    'controller_damping': 1000,
    'march_fn': march_soft
}

class Pusher:
    def __init__(self, options = OPTIONS):
        self.options = options
        pygame.init()
        self.screen = pygame.display.set_mode(self.options['screen_size'])
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = self.draw_options.DRAW_SHAPES | self.draw_options.DRAW_COLLISION_POINTS
    
        self.space = pymunk.Space()
        self.space.damping = self.options['damping']

        self.IMG_FLAG = self.options['block_img_flag']

        self._setup_env()
        
        if self.IMG_FLAG:
            self._setup_block_img()
            self._setup_target_img()
        else:
            # self._setup_block()
            self._setup_block_lines()
            self._setup_target()
        
        self._setup_pusher()

    def _setup_env(self):
        pts = self.options['boundary_pts']
        for i in range(len(pts)):
            seg = pymunk.Segment(self.space.static_body, pts[i], pts[(i+1)%len(pts)], 2)
            seg.elasticity = 0.999
            self.space.add(seg)

    def _setup_block(self):
        pts = self.options['block_pts']
        mass = self.options['block_mass']
        init_pos = Vec2d(*self.options['block_start_pos'])
        moment = pymunk.moment_for_poly(mass, pts)
        self.block_body = pymunk.Body(mass, moment)
        self.block_body.position = init_pos
        self.block_shape = pymunk.Poly(self.block_body, pts)
        self.block_shape.elasticity = self.options['elasticity']
        self.block_shape.friction = self.options['friction']
        self.block_shape.color = self.options['block_color']
        self.space.add(self.block_body, self.block_shape)

    def _setup_block_lines(self):
        pts = self.options['block_pts']
        mass = self.options['block_mass']
        init_pos = Vec2d(*self.options['block_start_pos'])
        moment = pymunk.moment_for_poly(mass, pts)
        self.block_body = pymunk.Body(mass, moment)
        self.block_body.position = init_pos
        self.block_shape = pymunk.Poly(self.block_body, pts)

        self.space.add(self.block_body)
        for i in range(len(pts)):
            seg = pymunk.Segment(self.block_body, pts[i], pts[(i+1)%len(pts)], 1)
            seg.elasticity = self.options['elasticity']
            seg.friction = self.options['friction']
            seg.color = self.options['block_color']
            self.space.add(seg)

    def _norm_pt(self, P, len_x, len_y):
        pt = P.x - (len_x - 1.0)/2.0, P.y - (len_y - 1.0)/2.0
        return Vec2d(*pt)

    def _setup_block_img(self):
        img = self.options['block_img']
        len_x = len(img)
        len_y = len(img[0])
        scale = self.options['block_img_scale']
        def sample_func(point):
            x = int(point[0])
            y = int(point[1])
            return 1 if img[x][y] == "x" else 0
        pl_set = self.options['march_fn'](pymunk.BB(0,0,len_x-1,len_y-1), len_x, len_y, .5, sample_func)
        edge_set = []
        for poly_line in pl_set:
            for i in range(len(poly_line) - 1):
                a = self._norm_pt(poly_line[i], len_x, len_y)
                edge_set.append(scale*a)

        mass = self.options['block_mass']
        init_pos = Vec2d(*self.options['block_start_pos'])
        moment = pymunk.moment_for_poly(mass, edge_set)
        self.block_body = pymunk.Body(mass, moment)
        self.block_body.position = init_pos

        self.space.add(self.block_body)
        for poly_line in pl_set:
            for i in range(len(poly_line) - 1):
                a = self._norm_pt(poly_line[i], len_x, len_y)
                b = self._norm_pt(poly_line[i + 1], len_x, len_y)
                seg = pymunk.Segment(self.block_body, scale*a, scale*b, 1)
                seg.elasticity = self.options['elasticity']
                seg.friction = self.options['friction']
                seg.color = self.options['block_color']
                self.space.add(seg)


    def _setup_pusher(self):
        init_pos = Vec2d(*self.options['pusher_start_pos'])
        mass = self.options['pusher_mass']
        radius = self.options['pusher_radius']
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))

        self.key_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.key_body.position = init_pos
        self.key_shape = pymunk.Circle(self.key_body, 0.01, (0, 0))
        self.key_shape.filter = pymunk.ShapeFilter(categories=0, mask=0)
        self.space.add(self.key_body, self.key_shape)

        self.push_body = pymunk.Body(mass, moment)
        self.push_body.position = init_pos
        self.push_shape = pymunk.Circle(self.push_body, radius, (0, 0))
        self.push_shape.elasticity = self.options['elasticity']
        self.push_shape.friction = self.options['friction']
        self.push_shape.color = self.options['pusher_color']
        self.space.add(self.push_body, self.push_shape)

        c = pymunk.constraints.DampedSpring(self.key_body, self.push_body, 
            anchor_a = (0,0), anchor_b = (0,0), rest_length=0.0, 
            stiffness=self.options['controller_stiffness'], 
            damping=self.options['controller_damping'])
        self.space.add(c)

    def _setup_target(self):
        pts = self.options['block_pts']
        pos = Vec2d(*self.options['target_start_pos'])
        moment = pymunk.moment_for_circle(0.1, 0, 0.1, (0, 0))
        self.target_body = pymunk.Body(0.1, moment)
        self.target_body.position = pos
        self.target_body.angle = self.options['target_start_angle']

        self.space.add(self.target_body)
        for i in range(len(pts)):
            seg = pymunk.Segment(self.target_body, pts[i], pts[(i+1)%len(pts)], 1)
            seg.filter = pymunk.ShapeFilter(categories=0, mask=0)
            seg.color = self.options['target_color']
            self.space.add(seg)

    def _setup_target_img(self):
        img = self.options['block_img']
        mass = self.options['block_mass']
        pos = Vec2d(*self.options['target_start_pos'])
        moment = pymunk.moment_for_circle(0.1, 0, 0.1, (0, 0))
        self.target_body = pymunk.Body(mass, moment)
        self.target_body.position = pos
        self.target_body.angle = self.options['target_start_angle']
        self.space.add(self.target_body)

        len_x = len(img)
        len_y = len(img[0])
        scale = self.options['block_img_scale']
        def sample_func(point):
            x = int(point[0])
            y = int(point[1])
            return 1 if img[x][y] == "x" else 0
        pl_set = self.options['march_fn'](pymunk.BB(0,0,len_x-1,len_y-1), len_x, len_y, .5, sample_func)

        for poly_line in pl_set:
            for i in range(len(poly_line) - 1):
                a = poly_line[i]
                b = poly_line[i + 1]
                seg = pymunk.Segment(self.target_body, scale*a, scale*b, 1)
                seg.filter = pymunk.ShapeFilter(categories=0, mask=0)
                seg.color = self.options['target_color']
                self.space.add(seg)


    def step(self, action):
        dx = action[0]
        dy = action[1]
        curx = self.key_body.position.x
        cury = self.key_body.position.y
        self.key_body.position = Vec2d(curx + dx, cury + dy)
        self.space.step(self.options['dt'])
        
    def render(self):
        self.screen.fill(pygame.Color("white"))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(50)

import shapely.geometry as sg
from shapely.ops import unary_union, polygonize

def pymunk_to_shapely(body, shapes):
    polygons = []
    segment_lines = []
    
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts.append(verts[0])  # Ensure the polygon is closed
            polygons.append(sg.Polygon(verts))
        elif isinstance(shape, pymunk.shapes.Segment):
            a = body.local_to_world(shape.a)
            b = body.local_to_world(shape.b)
            segment_lines.append(sg.LineString([a, b]))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")

    poly_from_segments = list(polygonize(segment_lines))

    all_polys = polygons + poly_from_segments
    
    if not all_polys:
        raise RuntimeError("No valid polygon could be created from the shapes")

    if len(all_polys) == 1:
        return all_polys[0]

    return unary_union(all_polys)





class PushEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)
    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=False,
        render_size=224,
        reset_to_state=None,
        relative=True,
        action_scale=100,
        with_velocity=False,
        with_target=True,
        shape=X_img,
        color="LightSlateGray",
        options=OPTIONS
    ):  
        self.shape = shape
        self.color = color
        self._seed = None
        self.seed()
        self.window_size = ws = 600  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 150

        self.legacy = legacy
        self.relative = relative  # relative action space
        self.action_scale = action_scale

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float64),
            high=np.array([ws, ws, ws, ws, np.pi * 2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64,
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float64),
            high=np.array([ws, ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action
        

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None

        self.with_velocity = with_velocity
        self.with_target = with_target
        self.reset_to_state = reset_to_state
        self.coverage_arr = []

        with open(self.shape, 'rb') as f:
            shapes = pickle.load(f)
        
        shape = shapes[0]

        self.options = options
        self.options['block_img']=block_img_dict[shape]
        self.pusher = Pusher(self.options)

        self.screen = self.pusher.screen


    def reset(self):
        self._setup()
        state = self.reset_to_state
        if state is None:
            rs = self.random_state
            state = np.array(
                [
                    rs.randint(50, 450),
                    rs.randint(50, 450),
                    rs.randint(100, 400),
                    rs.randint(100, 400),
                    rs.randn() * 2 * np.pi - np.pi,
                ]
            )
        self._set_state(state)

        self.coverage_arr = []
        state = self._get_obs()
        visual = self._render_frame("rgb_array")
        proprio = state[:2]
        observation = {
            "visual": visual,
            "proprio": proprio
        }
        return observation, state
    
    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_pusher_x,pos_pusher_y = state[0],state[1]
        pos_block_x,pos_block_y = state[2],state[3]
        rot_block = state[4]
        pusher=self.pusher
        pusher.block_body.position = Vec2d(pos_block_x, pos_block_y)
        pusher.block_body.angle = rot_block
        pusher.push_body.position = Vec2d(pos_pusher_x, pos_pusher_y)

        pusher.key_body.position = Vec2d(pos_pusher_x, pos_pusher_y)

        pusher.space.step(1.0 / self.sim_hz)
        self.pusher=pusher


    def _get_obs(self):
        pusher_pos = np.array([self.pusher.push_body.position.x, self.pusher.push_body.position.y])
        block_pos = np.array([self.pusher.block_body.position.x, self.pusher.block_body.position.y])
        block_angle = np.array([self.pusher.block_body.angle %  (2 * np.pi)])
        
        obs = np.concatenate([pusher_pos,block_pos,block_angle])
        return obs
    
    def _render_frame(self,mode):
        pygame.event.pump()
        pusher=self.pusher

        pusher.render()

        frame = pygame.surfarray.array3d(pusher.screen)

        frame_cv = np.transpose(frame, (1, 0, 2))
        img = cv2.resize(frame_cv, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8 / 96 * self.render_size)
                thickness = int(1 / 96 * self.render_size)
                cv2.drawMarker(
                    img,
                    coord,
                    color=(255, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size,
                    thickness=thickness,
                )
        return img
    
    def step(self,action):
        dt = 1.0/self.sim_hz
        pusher = self.pusher

        dx = np.array(action[0])*self.action_scale
        dy = np.array(action[1])*self.action_scale

        action = np.array(action) * self.action_scale
        if self.relative:
            self.latest_action=np.array([action[0]+pusher.push_body.position[0],action[1]+pusher.push_body.position[1]])
        curx = pusher.key_body.position.x
        cury = pusher.key_body.position.y
        pusher.key_body.position = Vec2d(curx + dx, cury + dy)
        pusher.space.step(dt)

        goal_geom = pymunk_to_shapely(pusher.target_body, pusher.target_body.shapes)
        block_geom = pymunk_to_shapely(pusher.block_body, pusher.block_body.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = False  # coverage > self.success_threshold

        self.coverage_arr.append(coverage)

        state = self._get_obs()
        visual = self._render_frame("rgb_array")

        proprio = state[:2]

        observation = {
            "visual": visual,
            "proprio": proprio
        }

        info = self._get_info()
        info["state"] = state
        info["max_coverage"] = max(self.coverage_arr)
        info["final_coverage"] = self.coverage_arr[-1]

        self.pusher = pusher

        return observation, reward, done, info
    
    def _setup(self):

        self.goal_pose = np.array([256, 256, np.pi / 4])

        pusher = self.pusher
        pusher.target_body.position = Vec2d(256,256)
        pusher.target_body.angle = np.pi / 4

        self.pusher = pusher

        self.max_score = 50 * 100
        self.success_threshold = 0.95  # 95% coverage.

    

    def set_task_goal(self, goal):
        self.goal_pose = goal

    
    def _get_info(self):

        pusher=self.pusher
        info = {
            "pos_agent": np.array(pusher.push_body.position),
            "vel_agent": np.array([0.0,0.0]),
            "block_pose": np.array(list(pusher.block_body.position) + [pusher.block_body.angle]),
            "goal_pose": self.goal_pose,
        }
        return info
    
    def render(self, mode):
        return self._render_frame(mode)
    

    def close(self):
        pusher = self.pusher
        if pusher.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        self.random_state = np.random.RandomState(seed)
            
            