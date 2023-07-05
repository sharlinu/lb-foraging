import argparse
import logging
import random
import time
import gym
import numpy as np
from lbforaging.foraging.environment import ForagingEnv
from pyglet.window import key

logger = logging.getLogger(__name__)

class InteractivePolicy:
    def __init__(self, env):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move_a = [False for i in range(5)]
        self.move_b = [False for i in range(5)]
        # register keyboard events with this environment's window
        env.viewer.window.on_key_press = self.key_press
        env.viewer.window.on_key_release = self.key_release
        self.pressed = False
    def action(self, obs):
        # ignore observation and just act based on keyboard events
        u = 5
        if self.move_a[0]: u = 1
        if self.move_a[1]: u = 2
        if self.move_a[2]: u = 3
        if self.move_a[3]: u = 4
        if self.move_a[4]: u = 5

        v = 5
        if self.move_b[0]: v = 1
        if self.move_b[1]: v = 2
        if self.move_b[2]: v = 3
        if self.move_b[3]: v = 4
        if self.move_b[4]: v = 5

        return np.array([u, v])

    # keyboard event callbacks
    def key_press(self, k, mod):
        # print(k)
        # print('pressing', k)
        self.pressed = True
        if k==key.UP:  self.move_a[0] = True
        if k==key.DOWN: self.move_a[1] = True
        if k==key.LEFT:    self.move_a[2] = True
        if k==key.RIGHT:  self.move_a[3] = True
        if k==key.L : self.move_a[4] = True

        if k==key.W:  self.move_b[0] = True
        if k==key.S: self.move_b[1] = True
        if k==key.A:    self.move_b[2] = True
        if k==key.D:  self.move_b[3] = True
        if k==key.X : self.move_b[4] = True
        return True

    def key_release(self, k, mod):
        self.pressed = False
        if k==key.UP:  self.move_a[0] = False
        if k==key.DOWN: self.move_a[1] = False
        if k==key.LEFT:    self.move_a[2] = False
        if k==key.RIGHT:  self.move_a[3] = False
        if k==key.L : self.move_a[4] = False

        if k==key.W: self.move_b[0] = False
        if k==key.S: self.move_b[1] = False
        if k==key.A: self.move_b[2] = False
        if k==key.D: self.move_b[3] = False
        if k==key.X: self.move_b[4] = False
        # return True

def _game_loop(env, render):
    """
    """
    obs = env.reset()
    done = False

    if render:
        env.render()
        # time.sleep(0.5)
    policy = InteractivePolicy(env)

    while not done:
        # while policy.key_release
        actions = policy.action(obs)
        time.sleep(1)
        print(actions)
        nobs, nreward, ndone, _ = env.step(actions)
        if sum(nreward) > 0:
            print(nreward)

        if render:
            env.render()
            # time.sleep(0.5)

        done = np.all(ndone)
    # print(env.players[0].score, env.players[1].score)

def main(episodes=1, render=True):
    env = ForagingEnv(
        players=2,
        max_player_level=3,
        field_size=(6, 6),
        max_food=2,
        grid_observation=True,
        sight=6,
        max_episode_steps=10000000,
        force_coop=True,
    )
    obs = env.reset()
    env.render()
    env.seed(4001)
    for episode in range(episodes):
        _game_loop(env, render=render)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Play the level foraging game.")
    #
    # parser.add_argument("--render", action="store_true")
    # parser.add_argument(
    #     "--times", type=int, default=1, help="How many times to run the game"
    # )
    #
    # args = parser.parse_args()
    main(10, True)
