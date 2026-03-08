from collections import deque
import numpy as np
import random

GRID_SIZE = 15

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

DIRECTIONS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class SnakeEnv:
    """Headless snake game environment with an RL-compatible interface."""

    def __init__(self, grid_size=GRID_SIZE, stack_size=1):
        self.grid_size = grid_size
        self.snake = []
        self.direction = RIGHT
        self.food = None
        self.score = 0
        self.done = False

        self.stack_size = stack_size
        self.frame_stack = deque(maxlen=self.stack_size)

    def reset(self):
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        self.direction = RIGHT
        self.food = None
        self._place_food()
        self.score = 0
        self.done = False

        state = self.get_state()

        self.frame_stack.clear()
        for _ in range(self.stack_size):
            self.frame_stack.append(state)

        return np.concatenate(self.frame_stack, axis=0)

    def _place_food(self):
        occupied = set(self.snake)

        free_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in occupied
        ]

        if free_cells:
            self.food = random.choice(free_cells)
        else:
            self.food = None

    def _stacked_obs(self):
        state = self.get_state()
        self.frame_stack.append(state)
        return np.concatenate(self.frame_stack, axis=0)

    def step(self, action):
        if self.done:
            return self._stacked_obs(), 0.0, True

        if action in DIRECTIONS and OPPOSITE.get(action) != self.direction:
            self.direction = action

        old_head = self.snake[0]

        old_distance = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])

        dr, dc = DIRECTIONS[self.direction]
        new_head = (old_head[0] + dr, old_head[1] + dc)

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.done = True
            return self._stacked_obs(), -10.0, True

        if new_head in self.snake[:-1]:
            self.done = True
            return self._stacked_obs(), -10.0, True

        self.snake.insert(0, new_head)

        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        reward = -0.01

        if new_distance < old_distance:
            reward += 0.1
        elif new_distance > old_distance:
            reward -= 0.1

        if new_head == self.food:
            self.score += 1
            reward += 10.0
            self._place_food()
        else:
            self.snake.pop()

        return self._stacked_obs(), reward, False

    def get_state(self):
        state = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)

        for r, c in self.snake:
            state[0, r, c] = 1.0

        if self.food is not None:
            state[1, self.food[0], self.food[1]] = 1.0

        assert state.shape == (2, self.grid_size, self.grid_size)
        assert state[1].sum() == 1.0
        assert state[0].sum() == len(self.snake)

        return state


if __name__ == "__main__":
    env = SnakeEnv()
    state = env.reset()

    total_reward = 0.0
    steps = 0

    while not env.done:
        action = random.randint(0, 3)
        state, reward, done = env.step(action)

        total_reward += reward
        steps += 1

    print(f"Game over after {steps} steps")
    print(f"Score (food eaten): {env.score}")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Final snake length: {len(env.snake)}")
    print(f"Board shape: {state.shape}")