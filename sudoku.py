import random
import copy
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims=[256, 128, 128]):
        super(PPOActorCritic, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.actor = nn.Linear(hidden_dims[-1], action_dim)
        self.critic = nn.Linear(hidden_dims[-1], 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value
    
class PPOAgent:
    def __init__(self, input_dim, action_dim, gamma=0.99, epochs=3,
                  batchSize=32768, lr=0.0001, clipEpsilon=0.1, entCoef=0.05,
                  valCoef=0.5, memory=10000, gradClip=0.5):
        self.action_dim = action_dim
        self.gamma = gamma
        self.temperature = 0
        self.clipEpsilon = clipEpsilon
        self.entCoef = entCoef
        self.valCoef = valCoef
        self.gradClip = gradClip
        self.epochs = epochs
        self.batchSize = batchSize
        self.memory = deque(maxlen=memory)
        self.network = PPOActorCritic(input_dim, action_dim).to(device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr)
        self.steps = 0

    def softMax(self, value, dim=-1):
        if self.temperature:
            return torch.softmax(value / self.temperature, dim=dim)
        else:
            return torch.softmax(value, dim=dim)

    def tupleAction(self, actionID):
        i = actionID // 81
        remainder = actionID % 81
        j = remainder // 9
        value = (remainder % 9) + 1
        return i, j, value

    def act(self, state, grid=None):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_logits, _ = self.network(state)
            if grid is not None:
                mask = torch.ones(self.action_dim).to(device)
                for idx in range(self.action_dim):
                    i, j, value = self.tupleAction(idx)
                    if grid[i][j] != 0 or not isValid(grid, i, j, value):
                        mask[idx] = 0
                action_logits = action_logits.masked_fill(mask == 0, -999_999)
            probs = self.softMax(action_logits)
            actionID = torch.multinomial(probs, 1).item()
            return self.tupleAction(actionID)

    def remember(self, state, actionID, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_logits, value = self.network(state_tensor)
            probs = self.softMax(action_logits)
            log_prob = torch.log(probs[0, actionID] + 1e-8)
        self.memory.append((state_tensor, actionID, reward, next_state, done, log_prob, value))

    def train(self):
        if len(self.memory) < self.batchSize:
            return
        batch = random.sample(self.memory, min(len(self.memory), self.batchSize))
        states, actions, rewards, next_states, dones, old_log_probs, old_values = zip(*batch)
        states = torch.cat(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.cat([torch.FloatTensor(ns).unsqueeze(0) for ns in next_states]).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        old_log_probs = torch.stack(old_log_probs).to(device)
        old_values = torch.cat(old_values).squeeze().to(device)
        if next_states.nelement() > 0:
            with torch.no_grad():
                _, next_values = self.network(next_states)
            returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        else:
            returns = rewards
        returns = returns.detach()
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()
        total_loss = 0.0
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            action_logits, values = self.network(states)
            probs = self.softMax(action_logits)
            log_probs = torch.log(probs + 1e-8)
            dist_entropy = -(log_probs * probs).sum(-1).mean()
            new_log_prob = log_probs[range(self.batchSize), actions]
            ratio = torch.exp(new_log_prob - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clipEpsilon, 1 + self.clipEpsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            loss = policy_loss - self.entCoef * dist_entropy + self.valCoef * value_loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.gradClip)
            self.optimizer.step()
            self.steps += 1
        self.memory.clear()


def getState(grid, base):
    return np.array([grid, base]).flatten()

def isValid(grid, i, j, value):
    if value in grid[i, :] or value in grid[:, j]:
        return False
    boxRow, boxCol = 3 * (i // 3), 3 * (j // 3)
    if value in grid[boxRow:boxRow+3, boxCol:boxCol+3]:
        return False
    return True

def genSudoku(hints):
    def canPlace(grid, row, col, num):
        if num in grid[row]:
            return False
        for i in range(9):
            if grid[i][col] == num:
                return False
        boxRow = (row // 3) * 3
        boxCol = (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if grid[boxRow + i][boxCol + j] == num:
                    return False
        return True

    def findEmpty(grid):
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    return i, j
        return None

    def fillGrid(grid):
        pos = findEmpty(grid)
        if pos is None:
            return True
        row, col = pos
        numbers = list(range(1, 10))
        random.shuffle(numbers)
        for num in numbers:
            if canPlace(grid, row, col, num):
                grid[row][col] = num
                if fillGrid(grid):
                    return True
                grid[row][col] = 0
        return False

    def countSolutions(grid):
        tempGrid = copy.copy(grid)
        solutions = 0
        def solve(solutions, tempGrid):
            if solutions > 1:
                return
            pos = findEmpty(tempGrid)
            if pos is None:
                solutions += 1
                return
            row, col = pos
            for num in range(9):
                if canPlace(tempGrid, row, col, num+1):
                    tempGrid[row][col] = num+1
                    solve(solutions, tempGrid)
                    tempGrid[row][col] = 0
        solve(solutions, tempGrid)
        return solutions

    def removeNumbers(grid, n):
        positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(positions)
        removed = 0
        while removed < (81 - n) and positions:
            pos = positions.pop()
            row, col = pos
            backup = grid[row][col]
            grid[row][col] = 0
            grid_copy = copy.deepcopy(grid)
            num_solutions = countSolutions(grid_copy)
            if num_solutions == 1:
                removed += 1
            else:
                grid[row][col] = backup

        return grid

    grid = np.zeros((9,9), dtype=int)
    fillGrid(grid)
    grid = removeNumbers(grid, hints)
    print(grid)
    return grid

# Define sweep configuration
sweep_config = {
  "method": "bayes",
  "metric": {
    "name": "time",
    "goal": "minimize"
  },
  "parameters": {
    "gamma": {
      "values": [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999]
    },
    "epochs": {
      "values": [1, 3, 5, 7, 9]
    },
    "batchSize": {
      "values": [2**12, 2**13, 2**14, 2**15, 2**16, 2**17]
    },
    "entCoef": {
      "distribution": "uniform",
      "min": 0.01,
      "max": 0.2
    },
    "valCoef": {
      "distribution": "uniform",
      "min": 0.1,
      "max": 0.9
    },
    "clipEpsilon": {
      "values": [0.03, 0.06, 0.09, 0.12, 0.15]
    },
    "gradClip": {
      "values": [0.3, 0.4, 0.5, 0.6, 0.7]
    },
    "memory": {
      "values": [2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20]
    },
    "lr": {
      "distribution": "uniform",
      "min": 0.00003,
      "max": 0.002
    }
  }
}


def train():
    with wandb.init() as run:
        config = wandb.config
        agent = PPOAgent(input_dim=162, action_dim=729, gamma=config.gamma,
                        epochs=config.epochs, batchSize=config.batchSize, lr=config.lr,
                        entCoef=config.entCoef, valCoef=config.valCoef, clipEpsilon=config.clipEpsilon,
                        memory=config.memory, gradClip=config.gradClip
                        )
        hints = 80
        startTime = time.time()
        #timeLimit = startTime + 180 #secs


        for sol in range(5):
            random.seed(sol)
            grid = genSudoku(hints)
            base = copy.copy(grid)
            print(base)
            solved = False
            attempts = 0
            while not solved:
                grid = copy.copy(base)
                state = getState(grid, base)
                done = False
                score = 0
                numbers = hints

                while not done:
                    i, j, value = agent.act(state, base)
                    reward = 0

                    if base[i][j] == 0:
                        if isValid(grid, i, j, value):
                                grid[i][j] = value
                                attempts = 0
                                reward = 5 + (numbers//8)
                                numbers += 1
                                if numbers == 81:
                                    reward = 100
                                    done = True
                                    solved = True
                        else:
                            reward = -1
                            attempts += 1
                            if attempts >= 11:
                                done = True
                    else:
                        reward = -2

                    if score < -999:
                        done = True

                    next_state = getState(grid, base)
                    actionID = i * 81 + j * 9 + (value - 1)
                    agent.remember(state, actionID, reward, next_state, done)
                    agent.train()
                    state = next_state
                    score += reward
                    print(score)

            hints -= 1
            wandb.log({"score": score})
            wandb.log({"solves": sol})
            wandb.log({"time": (np.round(time.time() - startTime))})

sweep_id = wandb.sweep(sweep_config, project="sudoku_ppo")
wandb.agent(sweep_id, train, count=50)
