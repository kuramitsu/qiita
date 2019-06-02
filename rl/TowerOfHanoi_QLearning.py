# -*- coding: utf-8 -*-

"""
https://qiita.com/akih1992/items/cdb39e5a23dff9b13498
"""

import numpy as np
from collections import defaultdict


class TowerOfHanoiEnvironment(object):
    def __init__(self, n_disks, max_episode_steps=200, n_poles=3):
        assert(n_poles >= 3)
        self.n_disks = n_disks
        self.n_poles = n_poles
        self.n_actions = int(n_poles * (n_poles - 1) / 2)   # n_poles C 2
        self.max_episode_steps = max_episode_steps
        state = np.zeros(n_poles * n_disks, dtype=np.bool)
        state[:n_disks].fill(True)
        self.state = state
        self.curr_step = 0
        # 以下は算出できそうな気もする
        action_map = []
        for p1 in range(n_poles):
            for p2 in range(p1 + 1, n_poles):
                action_map.append((p1, p2))
        self.action_map = action_map
        assert(len(self.action_map) == self.n_actions)
 
    def reset(self):
        self.state.fill(False)
        self.state[:self.n_disks].fill(True)
        self.curr_step = 0
        return self.state
    
    def _pole_state(self, pole_id):
        _s = pole_id * self.n_disks
        return self.state[_s:_s + self.n_disks]

    def step(self, action):
        self.curr_step += 1
        result = self.move_disk(*self.action_map[action])
        if not result:  # NO_MOVE
            is_terminal = False
            reward = -2
            return self.state, reward, is_terminal

        is_terminal = False
        reward = -1
        if not np.any(self._pole_state(0)):
            for pole_id in range(1, self.n_poles):
                if np.all(self._pole_state(pole_id)):
                    is_terminal = True
                    reward = 1

        if self.curr_step >= self.max_episode_steps:
            is_terminal = True
        
        return self.state, reward, is_terminal

    def _pole_top_disk(self, pole_id):
        state = self._pole_state(pole_id)
        for disk_id in range(self.n_disks):
            if state[disk_id]:
                return disk_id
        return self.n_disks    # sentinel

    def move_disk(self, pole_1, pole_2):
        top_1 = self._pole_top_disk(pole_1)
        top_2 = self._pole_top_disk(pole_2)
        if top_1 > top_2:
            self._pole_state(pole_1)[top_2] = True
            self._pole_state(pole_2)[top_2] = False
            return 1    # MOVE TO POLE1
        elif top_1 < top_2:
            self._pole_state(pole_1)[top_1] = False
            self._pole_state(pole_2)[top_1] = True
            return -1   # MOVE TO POLE2
        else:
            return 0    # NO MOVE

    def render(self):
        for pole_id in range(self.n_poles):
            disks = np.where(self._pole_state(pole_id))[0]
            print(f"pole_{pole_id}: {list(reversed(disks))}")


class QLearning(object):
    """
    Params:
        alpha : learning rate
        gamma : discount rate
    """
    def __init__(self, env, actor, alpha=0.01, gamma=0.99):
        self.env = env
        self.actor = actor
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(lambda: [0 for _ in range(self.env.n_actions)])
        self.training_episode_count = 0

    def play_episode(self, train=True, display=False):
        if train:
            self.training_episode_count += 1
        state = tuple(self.env.reset())
        is_terminal = False
        while not is_terminal:
            q_values = self.q_table[state]
            if train:
                action = self.actor.act_with_exploration(q_values)
                _state_array, reward, is_terminal = self.env.step(action)
                next_state = tuple(_state_array)
                self.update(state, action, reward, is_terminal, next_state)
            else:
                action = self.actor.act_without_exploration(q_values)
                _state_array, reward, is_terminal = self.env.step(action)
                next_state = tuple(_state_array)
            if display:
                print('----')
                print('step:{}'.format(self.env.curr_step))
                self.env.render()
            state = next_state

    def update(self, state, action, reward, is_terminal, next_state):
        target = reward + (1 - is_terminal) * max(self.q_table[next_state])
        self.q_table[state][action] *= self.alpha
        self.q_table[state][action] += (1 - self.alpha) * target


class EpsilonGreedyActor(object):
    def __init__(self, epsilon=0.1, random_state=None):
        self.epsilon = epsilon
        self.random = np.random
        if random_state is not None:
            self.random.seed(random_state)

    def act_without_exploration(self, q_values):
        max_q = max(q_values)
        argmax_list = [
            action for action, q in enumerate(q_values)
            if q == max_q
        ]
        return self.random.choice(argmax_list)

    def act_with_exploration(self, q_values):
        if self.random.uniform(0, 1) < self.epsilon:
            actions = np.arange(len(q_values))
            return self.random.choice(actions)
        else:
            return self.act_without_exploration(q_values)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--n_disks", type=int, default=5,
                    help="num of disks (default: {default}s)")
    ap.add_argument("-s", "--max_episode_steps", type=int, default=200,
                    help="maxnum of episode_steps (default: {default}s)")
    ap.add_argument("-e", "--n_episodes", type=int, default=200,
                    help="num of episodes (default: {default}s)")
    ap.add_argument("-p", "--n_poles", type=int, default=3,
                    help="num of poles (default: {default}s)")
    ap.add_argument("-P", "--plot", action="store_true")
    ap.add_argument("-D", "--display", action="store_true")
    args = ap.parse_args()

    n_disks = args.n_disks
    n_poles = args.n_poles
    env = TowerOfHanoiEnvironment(
        n_disks=n_disks,
        max_episode_steps=args.max_episode_steps,
        n_poles=n_poles)
    actor = EpsilonGreedyActor(random_state=0)
    model = QLearning(env, actor)
    n_episodes = args.n_episodes
    episode_steps_traj = []

    print('---- Start Training ----')
    for e in range(n_episodes):
        model.play_episode()
        episode_steps_traj.append(env.curr_step)
        if (e + 1) % 10 == 0:
            print('episode:{} episode_steps:{}'.format(
                model.training_episode_count,
                env.curr_step
            ))
    print('---- Finish Training ----')
    if args.display:
        env.reset()
        print("----")
        print("initial state")
        env.render()
        model.play_episode(train=False, display=True)
    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(np.arange(1, n_episodes + 1), episode_steps_traj, label='learning')
        if n_poles == 3:
            plt.plot([1, n_episodes + 1], [2**n_disks-1, 2**n_disks-1], label='shortest')
        plt.xlabel('episode')
        plt.ylabel('episode steps')
        plt.legend()
        plt.show()