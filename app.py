from flask import Flask, render_template, jsonify, request
import numpy as np

app = Flask(__name__)

class LabyrintheEnv:
    def __init__(self, size=8):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self._create_labyrinthe()
        self.reset()
        
    def _create_labyrinthe(self):
        self.grid[1, 1:4] = 1
        self.grid[2, 3] = 1
        self.grid[3, 1:4] = 1
        self.grid[3, 5:7] = 1
        self.grid[4, 2] = 1
        self.grid[4, 5] = 1
        self.grid[5, 2:5] = 1
        self.grid[6, 4] = 1
        self.grid[2, 5] = 4
        self.grid[5, 6] = 4
        self.grid[6, 2] = 4
        self.grid[0, 0] = 2
        self.grid[self.size-1, self.size-1] = 3
        
    def reset(self):
        self.agent_pos = [0, 0]
        self.done = False
        self.steps = 0
        return self._get_state()
    
    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def step(self, action):
        self.steps += 1
        new_pos = self.agent_pos.copy()
        if action == 0: new_pos[0] -= 1
        elif action == 1: new_pos[0] += 1
        elif action == 2: new_pos[1] -= 1
        elif action == 3: new_pos[1] += 1
        
        if new_pos[0] < 0 or new_pos[0] >= self.size or new_pos[1] < 0 or new_pos[1] >= self.size:
            reward = -1
            self.done = self.steps >= 100
            return self._get_state(), reward, self.done
        
        cell = self.grid[new_pos[0], new_pos[1]]
        
        if cell == 1:
            reward = -1
            self.done = self.steps >= 100
            return self._get_state(), reward, self.done
        
        self.agent_pos = new_pos
        
        if cell == 3:
            reward = 10
            self.done = True
        elif cell == 4:
            reward = -5
            self.done = True
        else:
            reward = -0.1
            self.done = self.steps >= 100
        
        return self._get_state(), reward, self.done
    
    def get_grid_for_display(self):
        display = self.grid.copy().tolist()
        display[self.agent_pos[0]][self.agent_pos[1]] = 5
        return display

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.n_actions = n_actions
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, episodes=500):
        self.history = []
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            self.decay_epsilon()
            self.history.append({
                'episode': ep + 1,
                'reward': total_reward,
                'steps': env.steps,
                'epsilon': round(self.epsilon, 4)
            })
        return self.history

env = LabyrintheEnv(size=8)
agent = QLearningAgent(n_states=64, n_actions=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/grid')
def get_grid():
    return jsonify({
        'grid': env.get_grid_for_display(),
        'size': env.size,
        'agent_pos': env.agent_pos
    })

@app.route('/api/train', methods=['POST'])
def train():
    data = request.get_json() or {}
    episodes = data.get('episodes', 500)
    alpha = data.get('alpha', 0.1)
    gamma = data.get('gamma', 0.9)
    epsilon = data.get('epsilon', 1.0)
    
    global agent
    agent = QLearningAgent(n_states=64, n_actions=4, alpha=alpha, gamma=gamma, epsilon=epsilon)
    history = agent.train(env, episodes=episodes)
    
    return jsonify({
        'success': True,
        'history': history,
        'final_epsilon': agent.epsilon,
        'q_table': agent.q_table.tolist()
    })

@app.route('/api/qtable')
def get_qtable():
    return jsonify({
        'q_table': agent.q_table.tolist(),
        'size': env.size
    })

@app.route('/api/play', methods=['POST'])
def play():
    state = env.reset()
    trajectory = [env.agent_pos.copy()]
    total_reward = 0
    done = False
    steps = 0
    max_steps = 50
    
    while not done and steps < max_steps:
        action = np.argmax(agent.q_table[state])
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1
        trajectory.append(env.agent_pos.copy())
    
    return jsonify({
        'trajectory': trajectory,
        'total_reward': total_reward,
        'steps': steps,
        'success': env.grid[env.agent_pos[0], env.agent_pos[1]] == 3
    })

@app.route('/api/reset')
def reset_env():
    env.reset()
    return jsonify({'success': True, 'grid': env.get_grid_for_display()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)