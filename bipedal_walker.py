import sys
import math

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

class GameConfig(object):
    FPS    = 50
    SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
    FRICTION = 2.5

    VIEWPORT_W = 600
    VIEWPORT_H = 400

class AgentConfig(object):
    LEG_DOWN_COEF = 8
    LEG_W_COEF = 8
    LEG_H_COEF = 34
    HULL_POLY = [
        (-30,+9), (+6,+9), (+34,+1),
        (+34,-8), (-30,-8)
    ]

    MOTORS_TORQUE = 80
    SPEED_HIP     = 4
    SPEED_KNEE    = 6
    LIDAR_RANGE_COEF = 160
    # LIDAR_RANGE   = 160/SCALE

    @classmethod
    def random(cls, seed):
        rng, seed = seeding.np_random(seed)
        return cls(
            LEG_DOWN_COEF=rng.randint(LEG_DOWN_COEF / 2, LEG_DOWN_COEF * 2 + 1),
            LEG_W_COEF=rng.randint(LEG_W_COEF / 2, LEG_W_COEF * 2 + 1),
            LEG_H_COEF=rng.randint(LEG_H_COEF / 2, LEG_H_COEF * 2 + 1),
            # HULL_POLY=???
            MOTORS_TORQUE=rng.randint(MOTORS_TORQUE / 2, MOTORS_TORQUE * 2 + 1),
            SPEED_HIP=rng.randint(SPEED_HIP / 2, SPEED_HIP * 2 + 1),
            SPEED_KNEE=rng.randint(SPEED_KNEE / 2, SPEED_KNEE * 2 + 1),
            LIDAR_RANGE_COEF=rng.randint(LIDAR_RANGE_COEF / 2, LIDAR_RANGE_COEF * 2 + 1),
        )


    def __init__(self, LEG_DOWN_COEF=LEG_DOWN_COEF, LEG_W_COEF=LEG_W_COEF, LEG_H_COEF=LEG_H_COEF,
                 MOTORS_TORQUE=MOTORS_TORQUE, SPEED_HIP=SPEED_HIP, SPEED_KNEE=SPEED_KNEE,
                 LIDAR_RANGE_COEF=LIDAR_RANGE_COEF, HULL_POLY=HULL_POLY):
        self.LEG_DOWN = -LEG_DOWN_COEF / GameConfig.SCALE
        self.LEG_W = LEG_W_COEF / GameConfig.SCALE
        self.LEG_H = LEG_H_COEF / GameConfig.SCALE
        self.MOTORS_TORQUE = MOTORS_TORQUE
        self.SPEED_HIP = SPEED_HIP
        self.SPEED_KNEE = SPEED_KNEE
        self.LIDAR_RANGE = LIDAR_RANGE_COEF / GameConfig.SCALE
        self.HULL_POLY = HULL_POLY

        self.HULL_FD = fixtureDef(
            shape=polygonShape(vertices=[ (x/GameConfig.SCALE, y/GameConfig.SCALE) for x, y in self.HULL_POLY ]),
            density=5.0,
            friction=0.1,
            categoryBits=0x0020,
            maskBits=0x001,   # collide only with ground
            restitution=0.0)  # 0.99 bouncy

        self.LEG_FD = fixtureDef(
            shape=polygonShape(box=(self.LEG_W/2, self.LEG_H/2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x0020,
            maskBits=0x001)

        self.LOWER_FD = fixtureDef(
            shape=polygonShape(box=(0.8*self.LEG_W/2, self.LEG_H/2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x0020,
            maskBits=0x001)

class EnvConfig(object):
    INITIAL_FORCE_RANDOM = 5

    TERRAIN_STEP_COEF = 14
    # TERRAIN_STEP   = 14/SCALE
    TERRAIN_LENGTH = 200     # in steps
    TERRAIN_HEIGHT_COEF = GameConfig.VIEWPORT_H/4
    # TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
    TERRAIN_GRASS    = 10    # low long are grass spots, in steps
    TERRAIN_STARTPAD = 20    # in steps

    PIT_LENGTH_MIN = 3
    PIT_LENGTH_MAX = 4  # inclusive

    STUMP_LENGTH_MIN = 1
    STUMP_LENGTH_MAX = 2  # inclusive

    STAIR_HEIGHT_ABS = 1
    STAIR_WIDTH_MIN = 4
    STAIR_WIDTH_MAX = 4  # inclusive
    STAIR_STEPS_MIN = 3
    STAIR_STEPS_MAX = 4  # inclusive

    @classmethod
    def random(cls, seed):
        rng, seed = seeding.np_random(seed)
        return cls(
            LEG_DOWN_COEF=rng.randint(LEG_DOWN_COEF / 2, LEG_DOWN_COEF * 2 + 1),
            LEG_W_COEF=rng.randint(LEG_W_COEF / 2, LEG_W_COEF * 2 + 1),
            LEG_H_COEF=rng.randint(LEG_H_COEF / 2, LEG_H_COEF * 2 + 1),
            MOTORS_TORQUE=rng.randint(MOTORS_TORQUE / 2, MOTORS_TORQUE * 2 + 1),
            SPEED_HIP=rng.randint(SPEED_HIP / 2, SPEED_HIP * 2 + 1),
            SPEED_KNEE=rng.randint(SPEED_KNEE / 2, SPEED_KNEE * 2 + 1),
            TERRAIN_STEP_COEF=rng.randint(TERRAIN_STEP_COEF / 2, TERRAIN_STEP_COEF * 2 + 1),
            LIDAR_RANGE_COEF=rng.randint(LIDAR_RANGE_COEF / 2, LIDAR_RANGE_COEF * 2 + 1),
        )

    def __init__(self, INITIAL_FORCE_RANDOM=INITIAL_FORCE_RANDOM,
                 TERRAIN_STEP_COEF=TERRAIN_STEP_COEF, TERRAIN_HEIGHT_COEF=TERRAIN_HEIGHT_COEF,
                 PIT_LENGTH_MIN=PIT_LENGTH_MIN, PIT_LENGTH_MAX=PIT_LENGTH_MAX,
                 STUMP_LENGTH_MIN=STUMP_LENGTH_MIN, STUMP_LENGTH_MAX=STUMP_LENGTH_MAX,
                 STAIR_HEIGHT_ABS=STAIR_HEIGHT_ABS, STAIR_WIDTH_MIN=STAIR_WIDTH_MIN, STAIR_WIDTH_MAX=STAIR_WIDTH_MAX,
                 STAIR_STEPS_MIN=STAIR_STEPS_MIN, STAIR_STEPS_MAX=STAIR_STEPS_MAX):
        self.INITIAL_FORCE_RANDOM = INITIAL_FORCE_RANDOM
        self.TERRAIN_STEP = TERRAIN_STEP_COEF / GameConfig.SCALE
        self.TERRAIN_HEIGHT = TERRAIN_HEIGHT_COEF / GameConfig.SCALE


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.hull == contact.fixtureA.body or self.env.hull == contact.fixtureB.body:
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False


class BipedalWalker(gym.Env, EzPickle):
    def __init__(self, hardcore=False, fix_terrain=False, fix_init_force=False,
                 agent_kwargs=dict(), env_kwargs=dict()):
        EzPickle.__init__(self)
        self.hardcore = hardcore
        self.fix_terrain = fix_terrain
        self.fix_init_force = fix_init_force
        self.agent_config = AgentConfig(**agent_kwargs)
        self.env_config = EnvConfig(**env_kwargs)
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : GameConfig.FPS
        }
        self.seed()

        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.cloud_poly = None
        self.hull = None
        self.init_force = None
        self.legs = []
        self.joints = []

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = GameConfig.FRICTION)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = GameConfig.FRICTION,
                    categoryBits=0x0001,
                )

        self.reset()

        high = np.array([np.inf] * 24)
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if self.terrain is not None and not self.fix_terrain:
            for t in self.terrain:
                self.world.DestroyBody(t)
            self.terrain = None
        if self.world.contactListener is not None:
            self.world.contactListener = None
        if self.hull is not None:
            self.world.DestroyBody(self.hull)
            self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _generate_terrain(self, hardcore):
        if self.terrain and self.fix_terrain:
            return

        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = self.env_config.TERRAIN_HEIGHT
        counter  = self.env_config.TERRAIN_STARTPAD
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(self.env_config.TERRAIN_LENGTH):
            x = i * self.env_config.TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(self.env_config.TERRAIN_HEIGHT - y)
                if i > self.env_config.TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1)/GameConfig.SCALE   #1

                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.randint(self.env_config.PIT_LENGTH_MIN, self.env_config.PIT_LENGTH_MAX + 1)
                poly = [
                    (x,                              y),
                    (x+self.env_config.TERRAIN_STEP, y),
                    (x+self.env_config.TERRAIN_STEP, y-4*self.env_config.TERRAIN_STEP),
                    (x,                              y-4*self.env_config.TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [(p[0] + self.env_config.TERRAIN_STEP * counter, p[1]) for p in poly]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4*self.env_config.TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.randint(self.env_config.STUMP_LENGTH_MIN, self.env_config.STUMP_LENGTH_MAX + 1)
                poly = [
                    (x,                                      y),
                    (x+counter*self.env_config.TERRAIN_STEP, y),
                    (x+counter*self.env_config.TERRAIN_STEP, y+counter*self.env_config.TERRAIN_STEP),
                    (x,                                      y+counter*self.env_config.TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                stair_height = +self.env_config.STAIR_HEIGHT_ABS if self.np_random.rand() > 0.5 else -self.env_config.STAIR_HEIGHT_ABS
                stair_width = self.np_random.randint(self.env_config.STAIR_WIDTH_MIN, self.env_config.STAIR_WIDTH_MAX + 1)
                stair_steps = self.np_random.randint(self.env_config.STAIR_STEPS_MIN, self.env_config.STAIR_STEPS_MAX + 1)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x+(    s*stair_width)*self.env_config.TERRAIN_STEP, y+(   s*stair_height)*self.env_config.TERRAIN_STEP),
                        (x+((1+s)*stair_width)*self.env_config.TERRAIN_STEP, y+(   s*stair_height)*self.env_config.TERRAIN_STEP),
                        (x+((1+s)*stair_width)*self.env_config.TERRAIN_STEP, y+(-1+s*stair_height)*self.env_config.TERRAIN_STEP),
                        (x+(    s*stair_width)*self.env_config.TERRAIN_STEP, y+(-1+s*stair_height)*self.env_config.TERRAIN_STEP),
                        ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                    t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                    self.terrain.append(t)
                counter = stair_steps*stair_width

            elif state == STAIRS and not oneshot:
                s = stair_steps*stair_width - counter - stair_height
                n = s/stair_width
                y = original_y + (n * stair_height) * self.env_config.TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.randint(self.env_config.TERRAIN_GRASS / 2, self.env_config.TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(self.env_config.TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (0.3, 1.0 if i%2 == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        if self.cloud_poly is not None and self.fix_terrain:
            return

        self.cloud_poly = []
        for i in range(self.env_config.TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, self.env_config.TERRAIN_LENGTH) * self.env_config.TERRAIN_STEP
            y = GameConfig.VIEWPORT_H/GameConfig.SCALE*3/4
            poly = [
                (x+15*self.env_config.TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*self.env_config.TERRAIN_STEP),
                 y+ 5*self.env_config.TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*self.env_config.TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly, x1, x2) )

    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        W = GameConfig.VIEWPORT_W / GameConfig.SCALE
        H = GameConfig.VIEWPORT_H / GameConfig.SCALE

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = self.env_config.TERRAIN_STEP*self.env_config.TERRAIN_STARTPAD/2
        init_y = self.env_config.TERRAIN_HEIGHT+2*self.agent_config.LEG_H
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            fixtures = self.agent_config.HULL_FD
                )
        self.hull.color1 = (0.5,0.4,0.9)
        self.hull.color2 = (0.3,0.3,0.5)

        if self.init_force is None or not self.fix_init_force:
            self.init_force = (self.np_random.uniform(-self.env_config.INITIAL_FORCE_RANDOM, self.env_config.INITIAL_FORCE_RANDOM), 0)
        self.hull.ApplyForceToCenter(self.init_force, True)

        self.legs = []
        self.joints = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position = (init_x, init_y - self.agent_config.LEG_H/2 - self.agent_config.LEG_DOWN),
                angle = (i*0.05),
                fixtures = self.agent_config.LEG_FD
                )
            leg.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            leg.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, self.agent_config.LEG_DOWN),
                localAnchorB=(0, self.agent_config.LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.agent_config.MOTORS_TORQUE,
                motorSpeed = i,
                lowerAngle = -0.8,
                upperAngle = 1.1,
                )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position = (init_x, init_y - self.agent_config.LEG_H*3/2 - self.agent_config.LEG_DOWN),
                angle = (i*0.05),
                fixtures = self.agent_config.LOWER_FD
                )
            lower.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            lower.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -self.agent_config.LEG_H/2),
                localAnchorB=(0, self.agent_config.LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.agent_config.MOTORS_TORQUE,
                motorSpeed = 1,
                lowerAngle = -1.6,
                upperAngle = -0.1,
                )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0
        self.lidar = [LidarCallback() for _ in range(10)]

        return self.step(np.array([0,0,0,0]))[0]

    def step(self, action):
        #self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(self.agent_config.SPEED_HIP  * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(self.agent_config.SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(self.agent_config.SPEED_HIP  * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(self.agent_config.SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed     = float(self.agent_config.SPEED_HIP     * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(self.agent_config.MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
            self.joints[1].motorSpeed     = float(self.agent_config.SPEED_KNEE    * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(self.agent_config.MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
            self.joints[2].motorSpeed     = float(self.agent_config.SPEED_HIP     * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(self.agent_config.MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
            self.joints[3].motorSpeed     = float(self.agent_config.SPEED_KNEE    * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(self.agent_config.MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

        self.world.Step(1.0/GameConfig.FPS, 6*30, 2*30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/10.0)*self.agent_config.LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/10.0)*self.agent_config.LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/GameConfig.FPS,
            0.3*vel.x*(GameConfig.VIEWPORT_W/GameConfig.SCALE)/GameConfig.FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(GameConfig.VIEWPORT_H/GameConfig.SCALE)/GameConfig.FPS,
            self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / self.agent_config.SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / self.agent_config.SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / self.agent_config.SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / self.agent_config.SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
            ]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 24

        self.scroll = pos.x - GameConfig.VIEWPORT_W/GameConfig.SCALE/5

        shaping  = 130*pos[0]/GameConfig.SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * self.agent_config.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (self.env_config.TERRAIN_LENGTH-self.env_config.TERRAIN_GRASS)*self.env_config.TERRAIN_STEP:
            done   = True
        return np.array(state), reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(GameConfig.VIEWPORT_W, GameConfig.VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, GameConfig.VIEWPORT_W/GameConfig.SCALE + self.scroll, 0, GameConfig.VIEWPORT_H/GameConfig.SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                                         0),
            (self.scroll+GameConfig.VIEWPORT_W/GameConfig.SCALE, 0),
            (self.scroll+GameConfig.VIEWPORT_W/GameConfig.SCALE, GameConfig.VIEWPORT_H/GameConfig.SCALE),
            (self.scroll,                                         GameConfig.VIEWPORT_H/GameConfig.SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2:
                continue
            if x1 > self.scroll/2 + GameConfig.VIEWPORT_W/GameConfig.SCALE:
                continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + GameConfig.VIEWPORT_W/GameConfig.SCALE:
                continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render+1) % 100
        i = self.lidar_render
        if i < 2*len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar)-i-1]
            self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = self.env_config.TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/GameConfig.SCALE
        x = self.env_config.TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/GameConfig.SCALE), (x+25/GameConfig.SCALE, flagy2-5/GameConfig.SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2)

        return self.viewer.render(return_rgb_array = mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = BipedalWalker()
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5*moving_leg
        supporting_s_base = 4 + 5*supporting_leg

        hip_targ  = [None,None]   # -0.8 .. +1.1
        knee_targ = [None,None]   # -0.6 .. +0.9
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state == STAY_ON_ONE_LEG:
            hip_targ[moving_leg]  = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED: supporting_knee_angle += 0.03
            supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state == PUT_OTHER_DOWN:
            hip_targ[moving_leg]  = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base+4]:
                state = PUSH_OFF
                supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
        if state == PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
        if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
        if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
        if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

        hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
        hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
        knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0*s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5*a, -1.0, 1.0)

        env.render()
        if done: break
