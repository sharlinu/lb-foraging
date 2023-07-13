import logging
import time
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env, spaces
import gym
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Player:
    def __init__(self, id=id):
        self.id = id
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        max_player_level,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        keep_food = False,
        simple = True,
        normalize_reward=True,
        grid_observation=False,
        penalty=0.0,
    ):
        self.logger = logging.getLogger(__name__)
        # self.seed()
        self.players = [Player(id=i) for i in range(players)]

        self.field = np.zeros(field_size, np.int32)
        self.keep_food = keep_food
        print('keep food', keep_food)
        self.simple = simple
        print('rationed food', simple)

        self.penalty = penalty
        
        self.max_food = max_food
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(6)] * len(self.players)))

        if not self._grid_observation:
            self.observation_space = spaces.Tuple(
                ([self._get_observation_space()] * len(self.players)))  # TODO could simplify this to one space?
        else:
            self.single_observation_space = spaces.Dict()
            self.single_observation_space['image'] = self._get_observation_space()
            self.observation_space = spaces.Tuple(
                ([self.single_observation_space] * len(self.players)))

        self.viewer = None

        self.n_agents = len(self.players)
        self.n_objects = self.max_food

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_food = self.max_food
            max_food_level = self.max_player_level * len(self.players)
            # max_food_level = 1

            min_obs = [-1, -1, 0] * max_food + [-1, -1, 0] * len(self.players)
            max_obs = [field_x-1, field_y-1, max_food_level] * max_food + [
                field_x-1,
                field_y-1,
                self.max_player_level,
            ] * len(self.players)
        else:
            # grid observation space
            # grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)


            # agents layer: agent levels
            agents_min = np.zeros(self.field_size, dtype=np.float32)
            agents_max = np.ones(self.field_size, dtype=np.float32) * self.max_player_level

            # foods layer: foods level
            max_food_level = self.max_player_level * len(self.players)
            # max_food_level = 1
            foods_min = np.zeros(self.field_size, dtype=np.float32)
            foods_max = np.ones(self.field_size, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(self.field_size, dtype=np.float32)
            access_max = np.ones(self.field_size, dtype=np.float32)

            # ego-layer: identity level
            ego_min = np.zeros(self.field_size, dtype=np.int64)
            ego_max = np.ones(self.field_size, dtype=np.int64)

            # total layer
            min_obs = np.stack([agents_min, ego_min, foods_min], axis=2)
            max_obs = np.stack([agents_max, ego_max, foods_max], axis=2) # TODO works well?

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in enumerate(obs.players):
            player = Player(id=p)
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        """
        Only returns food location when it is active, i.e. >0

        """
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self, max_food, max_level):
        food_count = 0
        attempts = 0
        min_level = max_level if self.force_coop else 1
        while food_count < max_food and attempts < 1000:
            attempts += 1
            row = self.np_random.randint(1, self.rows - 1)
            col = self.np_random.randint(1, self.cols - 1)
            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = (
                min_level
                if min_level == max_level
                # ! this is excluding food of level `max_level` but is kept for
                # ! consistency with prior LBF versions
                else self.np_random.randint(min_level, max_level)
            )
            food_count += 1
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] > 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, max_player_level):
        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.randint(0, self.rows)
                col = self.np_random.randint(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.np_random.randint(1, max_player_level + 1),
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_food):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            for i in range(len(self.players)):
                obs[self.max_food * 3 + 3 * i] = -1
                obs[self.max_food * 3 + 3 * i + 1] = -1
                obs[self.max_food * 3 + 3 * i + 2] = 0

            for i, p in enumerate(seen_players):
                obs[self.max_food * 3 + 3 * i] = p.position[0]
                obs[self.max_food * 3 + 3 * i + 1] = p.position[1]
                obs[self.max_food * 3 + 3 * i + 2] = p.level

            return obs

        def make_global_grid_arrays(observation):
            """
            Create global arrays for grid observation space
            """
            grid_shape_x, grid_shape_y = self.field_size
            # grid_shape_x += 2 * self.sight
            # grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)

            agents_layer = np.zeros(grid_shape, dtype=np.int64)
            ego_layer = np.zeros(grid_shape, dtype=np.int64)

            for player in self.players:
                player_x, player_y = player.position
                agents_layer[player_x, player_y] = player.level

            for player in observation.players:
                if player.is_self:
                    player_x, player_y = player.position
                    ego_layer[player_x, player_y] = 1
                    # print(f' Player at pos {player_x,player_y}')

            # foods_layer = np.zeros(grid_shape, dtype=np.float32)
            foods_layer = self.field.copy()


            return np.stack([agents_layer, ego_layer, foods_layer], axis=2)

        def get_agent_grid_bounds(agent_x, agent_y):
            return agent_x, agent_x + 2 * self.sight + 1, agent_y, agent_y + 2 * self.sight + 1
        
        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        observations = [self._make_obs(player) for player in self.players]
        if self._grid_observation:
            nobs = tuple([{'image': make_global_grid_arrays(obs)} for obs in observations])
            # layers = make_global_grid_arrays()
            # agents_bounds = [get_agent_grid_bounds(*player.position) for player in self.players]
            # nobs = tuple([layers[:, start_x:end_x, start_y:end_y] for start_x, end_x, start_y, end_y in agents_bounds])
        else:
            nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {}
        
        # check the space of obs
        # for i, obs in  enumerate(nobs):
        #     assert self.observation_space[i]['image'].contains(obs['image']), \
        #         f"obs space error: obs: {obs['image']}, obs_space: {self.observation_space[i]['image']}"
        
        return nobs, np.array(nreward), np.array(ndone), ninfo

    def reset(self):
        # print('seed:', self.seed())
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.max_player_level)
        player_levels = sorted([player.level for player in self.players])

        if self.simple:
            self.spawn_food(
                self.max_food, max_level=1)
        else:
            self.spawn_food(
                self.max_food, max_level=sum(player_levels[:2])
            )
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        nobs, _, _, _ = self._make_gym_obs()
        return nobs

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players
        sorted_keys = sorted(collisions, key=lambda k: len(collisions[k]), reverse=True)
        for k in sorted_keys:
            if len(collisions[k]) > 1:  # make sure no more than an player will arrive at location
                for a in collisions[k]:
                    if a.position != k :
                        collisions[a.position].append(a)
                continue
            # print(f'changing new position to {k}')
            collisions[k][0].position = k


        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])

            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                for a in adj_players:
                    a.reward -= self.penalty
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.reward = float(a.level * food)
                if self._normalize_reward:
                    a.reward = a.reward / float(
                        adj_player_level * self._food_spawned
                    )  # normalize reward
            # and the food is removed
            if self.keep_food:
                self.field[frow, fcol] = -1
            else:
                self.field[frow,fcol] = 0

        self._game_over = (
            np.max(self.field) <= 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        for p in self.players:
            p.score += p.reward

        return self._make_gym_obs()

    def _init_render(self):
        # from foraging.rendering import Viewer
        from lbforaging.foraging.rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()



def _game_loop(env, render=False):
    """
    """
    # env.seed(1)
    obs = env.reset()
    done = False
    if render:
        arr = env.render(mode='rgb_array')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(arr)
        # plt.axis('off')
        plt.savefig('nolines_run_initial.png')
        #pygame.display.update()  # update window
        time.sleep(0.5)
    ct =0
    while not done:

        actions = env.action_space.sample() # TODO this needs to be replaced
        # print('actions', actions)
        nobs, nreward, ndone, _ = env.step(actions)
        print('agent', nobs[0]['image'][:,:,0])
        print('ego', nobs[0]['image'][:, :, 1])
        print('food', nobs[0]['image'][:, :, 2])
        time.sleep(1)

        if sum(nreward) > 0:
            print(nreward)

        if render:
            # env.render(mode='rgb_array')
            arr = env.render(mode='rgb_array')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(arr)
            # plt.axis('off')
            plt.savefig(f'nolines_run_{ct}.png')

            #pygame.display.update()  # update window
            time.sleep(0.5)
        done = np.all(ndone)
        ct += 1
    print('episode finished')
    # print(env.players[0].score, env.players[1].score)



if __name__ == "__main__":
    env = ForagingEnv(
        players=2,
        max_player_level=3,
        field_size= (8,8),
        max_food=2,
        grid_observation=True,
        sight=8,
        max_episode_steps=10,
        force_coop=True,
    )
    # background_colour = (50,50,50)
    # obs = env.reset()
    # env.seed(4001)
    for episode in range(10):
        _game_loop(env,  render = True)
    # nobs, nreward, ndone, ninfo = env.step([1,1])
    print("Done")
