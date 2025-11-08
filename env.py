import time
import random
import gymnasium as gym
from enums.player_action import PlayerAction, PlayerActionTable
from game import Game

MOVEMENT_ACTIONS = [PlayerAction.NORTH, PlayerAction.SOUTH, PlayerAction.EAST, PlayerAction.WEST]


class SupermarketEnv(gym.Env):

    def __init__(self, num_players=1, player_speed=0.15, keyboard_input=False, render_messages=True, bagging=False,
                 headless=False, initial_state_filename=None, follow_player=-1, random_start=False,
                 render_number=False, max_num_items=33, player_sprites=None, record_path=None, stay_alive=False, mode=0, stochastic=False):

        super(SupermarketEnv, self).__init__()

        self.unwrapped.step_count = 0
        self.mode = mode
        self.render_messages = render_messages
        self.keyboard_input = keyboard_input
        self.render_number = render_number
        self.bagging = bagging
        self.stochastic =  stochastic

        self.follow_player = follow_player

        self.unwrapped.num_players = num_players
        self.player_speed = player_speed
        self.unwrapped.game= None
        self.player_sprites = player_sprites

        self.record_path = record_path
        self.unwrapped.max_num_items=max_num_items

        self.stay_alive = stay_alive

        self.initial_state_filename = initial_state_filename

        self.action_space = gym.spaces.Tuple([gym.spaces.Tuple((gym.spaces.Discrete(len(PlayerAction)),
                                                                gym.spaces.Discrete(max_num_items)))] * num_players)
        self.observation_space = gym.spaces.Dict()
        self.headless = headless
        self.random_start = random_start

        self.action_probability = {}
        
        if (stochastic): # storing probability of action success rate
            filename = 'stochastic_probability.txt'
            with open(filename, "r") as file: 
                content = file.read()
                for row in content.split("\n"):
                    action_row = list(map(lambda column: column.strip(": "), row.split("\t")))
                    probability_pairs = map(lambda result: tuple(result.split(" ")), action_row[1:])
                    self.action_probability[PlayerActionTable[action_row[0]]] = dict(map(lambda pair: (PlayerActionTable[pair[0]], float(pair[1])), probability_pairs))
        # else: 
        #     self.action_probability = Action_Probabilities

    def get_stochastic_action(self, action):
        return random.choices(list(self.action_probability[action].keys()), weights=list(self.action_probability[action].values()), k=1)[0]


    def step(self, action):
        done = False

        for i, player_action in enumerate(action):
            player_action, arg = player_action
            if self.stochastic == True:
                player_action = self.get_stochastic_action(player_action)
            if player_action in MOVEMENT_ACTIONS:
                self.unwrapped.game.player_move(i, player_action)
            elif player_action == PlayerAction.NOP:
                self.unwrapped.game.nop(i)
            elif player_action == PlayerAction.INTERACT:
                self.unwrapped.game.interact(i)
            elif player_action == PlayerAction.TOGGLE:
                self.unwrapped.game.toggle_cart(i)
                self.unwrapped.game.toggle_basket(i)
            elif player_action == PlayerAction.CANCEL:
                self.unwrapped.game.cancel_interaction(i)
            elif player_action == PlayerAction.PICKUP:
                self.unwrapped.game.pickup(i, arg)
            elif player_action == PlayerAction.RESET:
                self.reset()
        observation = self.unwrapped.game.observation()
        self.unwrapped.step_count += 1
        if not self.unwrapped.game.running:
            done = True
        return observation, 0., done, None, None

    def reset(self,seed = None, options = None, obs=None, mode = 0):
        self.unwrapped.game= Game(self.unwrapped.num_players, self.player_speed,
                         keyboard_input=self.keyboard_input,
                         render_messages=self.render_messages,
                         bagging=self.bagging,
                         headless=self.headless, initial_state_filename=self.initial_state_filename,
                         follow_player=self.follow_player, random_start=self.random_start,
                         render_number=self.render_number,
                         sprite_paths=self.player_sprites,
                         record_path=self.record_path,
                         stay_alive=self.stay_alive)
        self.unwrapped.game.set_up(mode=self.mode)
        if obs is not None:
            self.unwrapped.game.set_observation(obs)
        ########################
        # seed and options are added in gym v26, since seed() is removed from gym v26 and combined with reset(),
        # which are not currently used by the environment  
        if seed is not None:
            pass
        if options is not None:
            pass
        ########################
        self.unwrapped.step_count = 0
        return self.unwrapped.game.observation()

    def render(self, mode='human'):
        if mode.lower() == 'human' and not self.headless:
            self.unwrapped.game.update()
        #else:
        #   print(self.game.observation(True))


class SinglePlayerSupermarketEnv(gym.Wrapper):
    def __init__(self, env):
        super(SinglePlayerSupermarketEnv, self).__init__(env)
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.unwrapped.num_players),
                                              gym.spaces.Discrete(len(PlayerAction)),
                                              gym.spaces.Discrete(self.unwrapped.max_num_items)))

    def convert_action(self, player_action):
        i, action, arg = player_action
        full_action = [(PlayerAction.NOP, 0)]*self.unwrapped.num_players
        full_action[i] = (action, arg)
        return full_action

    def step(self, player_action):
        done = False
        i, player_action, arg = player_action
        if self.stochastic ==  True:
            player_action = self.get_stochastic_action(player_action)
        if player_action in MOVEMENT_ACTIONS:
            self.unwrapped.game.player_move(i, player_action)
        elif player_action == PlayerAction.NOP:
            self.unwrapped.game.nop(i)
        elif player_action == PlayerAction.INTERACT:
            self.unwrapped.game.interact(i)
        elif player_action == PlayerAction.TOGGLE:
            self.unwrapped.game.toggle_cart(i)
            self.unwrapped.game.toggle_basket(i)
        elif player_action == PlayerAction.CANCEL:
            self.unwrapped.game.cancel_interaction(i)
        elif player_action == PlayerAction.PICKUP:
            self.unwrapped.game.pickup(i, arg)
        elif player_action == PlayerAction.RESET:
            self.reset()
        observation = self.unwrapped.game.observation()
        self.unwrapped.step_count += 1
        if not self.unwrapped.game.running:
            done = True
        return observation, 0., done, None, None


if __name__ == "__main__":
    env = SupermarketEnv(2)
    env.reset()

    for i in range(100):
        env.step((PlayerAction.EAST, PlayerAction.SOUTH))
        env.render()
