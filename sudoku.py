import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

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
    def __init__(self, input_dim, action_dim):
        self.action_dim = action_dim
        self.gamma = 0.99
        self.temperature = 0
        self.clipEpsilon = 0.1
        self.entCoef = 0.04
        self.valCoef = 0.5
        self.gradClip = 1.0
        self.epochs = 3
        self.batchSize = 2**15
        self.memory = deque(maxlen=50000)
        self.network = PPOActorCritic(input_dim, action_dim).to(device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=0.0001)
        self.steps = 0

    def softMax(self, value, dim=-1):
        if self.temperature:
            return torch.softmax(value / self.temperature, dim=dim)
        else:
            return torch.softmax(value, dim=dim)


    def tupleAction(self, action_idx):
        """Convert action index (0-728) to tuple (i, j, value)."""
        i = action_idx // 81
        remainder = action_idx % 81
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

                action_logits = action_logits.masked_fill(mask == 0, -999999)

            probs = self.softMax(action_logits)
            action_idx = torch.multinomial(probs, 1).item()
            return self.tupleAction(action_idx)

    def remember(self, state, action_idx, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_logits, value = self.network(state_tensor)
            probs = self.softMax(action_logits)
            log_prob = torch.log(probs[0, action_idx] + 1e-8)

        self.memory.append((state_tensor, action_idx, reward, next_state, done, log_prob, value))

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


def createGrid(hints):
    grid = np.zeros((9, 9), dtype=int)
    slots = []
    for _ in range(hints):
        n = random.randint(0, 8)
        m = random.randint(0, 8)
        if [n, m] not in slots:
            slots.append([n, m])
            grid[n][m] = random.randint(1, 9)

    return grid

def hasSol(grid):
    row_used = [0] * 9
    col_used = [0] * 9
    box_used = [0] * 9
    
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                num = grid[i][j] - 1
                mask = 1 << num
                box_index = (i // 3) * 3 + (j // 3)
                if (row_used[i] & mask) or (col_used[j] & mask) or (box_used[box_index] & mask):
                    return False
                
                row_used[i] |= mask
                col_used[j] |= mask
                box_used[box_index] |= mask
    
    empty_cells = [(i, j) for i in range(9) for j in range(9) if grid[i][j] == 0]
    
    def backtrack(index):
        if index == len(empty_cells):
            return True
        
        i, j = empty_cells[index]
        box_index = (i // 3) * 3 + (j // 3)
        for num in range(1, 10):
            mask = 1 << (num - 1)
            if not (row_used[i] & mask) and not (col_used[j] & mask) and not (box_used[box_index] & mask):
                grid[i][j] = num
                row_used[i] |= mask
                col_used[j] |= mask
                box_used[box_index] |= mask
                if backtrack(index + 1):
                    return True
                
                grid[i][j] = 0
                row_used[i] &= ~mask
                col_used[j] &= ~mask
                box_used[box_index] &= ~mask

        return False
    
    return backtrack(0)

def createSolveable(lower, upper):
    while True:
        grid = createGrid(random.randint(lower, upper))
        q = hasSol(grid)
        if q:
            return grid

def getState(grid, base):
    return np.array([grid, base]).flatten()

def isValid(grid, i, j, value):
    if value in grid[i, :] or value in grid[:, j]:
        return False
    
    box_row, box_col = 3 * (i // 3), 3 * (j // 3)
    if value in grid[box_row:box_row+3, box_col:box_col+3]:
        return False
    
    return True

def evaluate(agent, num_tests=10):
    successes = 0
    for _ in range(num_tests):
        grid, solved = createSolveable()
        state = getState(grid, base)
        done = False
        while not done:
            i, j, value = agent.act(state, base)
            if grid[i][j] == 0:
                grid[i][j] = value
                state = getState(grid, base)
                if np.array_equal(grid, solved):
                    successes += 1
                    break

            else:
                break

    return successes / num_tests


def genSudoku(lower, upper):
    if lower < 0 or upper > 81 or lower > upper:
        raise ValueError("Invalid range for starting numbers")
    
    def canPlace(grid, row, col, num):
        if num in grid[row]:
            return False
        
        for i in range(9):
            if grid[i][col] == num:
                return False
            
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if grid[box_row + i][box_col + j] == num:
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
        grid_copy = [row[:] for row in grid]
        solutions = 0
        def solve():
            nonlocal solutions, grid_copy
            if solutions > 1:
                return
            
            pos = findEmpty(grid_copy)
            if pos is None:
                solutions += 1
                return
            
            row, col = pos
            for num in range(1, 10):
                if canPlace(grid_copy, row, col, num):
                    grid_copy[row][col] = num
                    solve()
                    grid_copy[row][col] = 0

        solve()
        return solutions

    def removeNumbers(grid, s):
        attempts = 81 - s
        positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(positions)
        removed = 0
        while removed < attempts and positions:
            pos = positions.pop()
            row, col = pos
            backup = grid[row][col]
            grid[row][col] = 0
            grid_copy = [row[:] for row in grid]
            num_solutions = countSolutions(grid_copy)
            if num_solutions == 1:
                removed += 1

            else:
                grid[row][col] = backup

        return grid

    s = random.randint(lower, upper)
    grid = np.array([[0 for _ in range(9)] for _ in range(9)])
    fillGrid(grid)
    grid = removeNumbers(grid, s)
    return grid

if __name__ == "__main__":
    agent = PPOAgent(162, 729)
    lower = 80
    upper = 80
    bestScore = 1
    sols = 0

    for ep in range(70):
        random.seed(ep)
        grid = genSudoku(lower, upper)
        base = copy.copy(grid)
        solved = False
        atts = 0
        sols += 1
        while not solved:
            atts += 1
            grid = copy.copy(base)
            state = getState(grid, base)
            done = False
            score = 0
            acts = 0
            attempts = 0
            numbers = lower

            while not done:
                i, j, value = agent.act(state, base)
                acts += 1
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
                        if attempts >= 9:
                            done = True
                else:
                    reward = -2

                if score < -999:
                    done = True

                next_state = getState(grid, base)
                action_idx = i * 81 + j * 9 + (value - 1)
                agent.remember(state, action_idx, reward, next_state, done)
                agent.train()
                state = next_state
                score += reward
            print(sols)

        lower -= 1
        upper -= 1
        #print(-atts)
