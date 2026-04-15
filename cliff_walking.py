import numpy as np
import matplotlib.pyplot as plt

class CliffWalkingEnv:
    """Cliff Walking 懸崖尋路環境"""
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        # 定義懸崖的位置
        self.cliff = [(3, i) for i in range(1, 11)]
        
        # 動作空間：0 上, 1 下, 2 左, 3 右
        self.actions = [0, 1, 2, 3]
        
    def reset(self):
        """重置環境狀態至起點"""
        self.state = self.start_state
        return self.state
    
    def step(self, action):
        """根據動作更新狀態，並回傳 (next_state, reward, done)"""
        i, j = self.state
        
        if action == 0:   # 上
            next_state = (max(i - 1, 0), j)
        elif action == 1: # 下
            next_state = (min(i + 1, self.rows - 1), j)
        elif action == 2: # 左
            next_state = (i, max(j - 1, 0))
        elif action == 3: # 右
            next_state = (i, min(j + 1, self.cols - 1))
            
        reward = -1
        done = False
        
        # 判斷是否掉入懸崖或抵達終點
        if next_state in self.cliff:
            reward = -100
            next_state = self.start_state # 掉入懸崖重置回起點，但不結束 episode
        elif next_state == self.goal_state:
            done = True
            
        self.state = next_state
        return next_state, reward, done

class RLAgent:
    """強化學習基礎 Agent"""
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha       # 學習率
        self.gamma = gamma       # 折扣因子
        self.epsilon = epsilon   # 探索率
        # 初始化 Q-table 為全 0
        self.q_table = np.zeros((env.rows, env.cols, len(env.actions)))
        
    def choose_action(self, state):
        """使用 epsilon-greedy 策略選擇動作"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            i, j = state
            values = self.q_table[i, j, :]
            # 處理多個最大值的情況，從中隨機選一個 (避免總是選同一個最高分動作)
            return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

class SarsaAgent(RLAgent):
    """SARSA 演算法 (On-policy)"""
    def update(self, state, action, reward, next_state, next_action):
        i, j = state
        next_i, next_j = next_state
        
        # SARSA 更新公式: Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
        # 這裡的 Q(S',A') 使用的是「實際」選擇的下一個動作 next_action 的值
        td_target = reward + self.gamma * self.q_table[next_i, next_j, next_action]
        td_error = td_target - self.q_table[i, j, action]
        self.q_table[i, j, action] += self.alpha * td_error

class QLearningAgent(RLAgent):
    """Q-learning 演算法 (Off-policy)"""
    def update(self, state, action, reward, next_state):
        i, j = state
        next_i, next_j = next_state
        
        # Q-learning 更新公式: Q(S,A) <- Q(S,A) + alpha * [R + gamma * max_a Q(S',a) - Q(S,A)]
        # 這裡的 target 直接使用下一狀態中「最大」的 Q 值，而不考量實際執行的行為
        td_target = reward + self.gamma * np.max(self.q_table[next_i, next_j, :])
        td_error = td_target - self.q_table[i, j, action]
        self.q_table[i, j, action] += self.alpha * td_error


def train_sarsa(episodes=500):
    env = CliffWalkingEnv()
    agent = SarsaAgent(env)
    rewards_history = []
    
    for _ in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            
            agent.update(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
            total_reward += reward
            
        rewards_history.append(total_reward)
        
    return agent, rewards_history

def train_q_learning(episodes=500):
    env = CliffWalkingEnv()
    agent = QLearningAgent(env)
    rewards_history = []
    
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
    return agent, rewards_history

def plot_path(agent, title):
    """顯示在沒有 epsilon 隨機探索情況下的最優路徑策略"""
    env = CliffWalkingEnv()
    state = env.reset()
    done = False
    
    # 建立網格地圖
    path = np.full((env.rows, env.cols), '.', dtype=str)
    for i, j in env.cliff:
        path[i, j] = 'C'
    path[env.start_state] = 'S'
    path[env.goal_state] = 'G'
    
    while not done:
        i, j = state
        # 單純取最大值當最優步驟 (greedy path)
        action = np.argmax(agent.q_table[i, j, :])
        
        if action == 0: path[i, j] = '↑'
        elif action == 1: path[i, j] = '↓'
        elif action == 2: path[i, j] = '←'
        elif action == 3: path[i, j] = '→'
        
        next_state, _, done = env.step(action)
        state = next_state
        if next_state in env.cliff or next_state == env.goal_state:
            break
            
    print(f"\nFinal greedy path for {title}:")
    for row in path:
        print(' '.join(row))


def main():
    episodes = 500
    
    print("Training SARSA...")
    sarsa_agent, sarsa_rewards = train_sarsa(episodes)
    
    print("Training Q-learning...")
    q_learning_agent, q_learning_rewards = train_q_learning(episodes)
    
    # 印出最終學習到的策略路徑
    plot_path(sarsa_agent, 'SARSA')
    plot_path(q_learning_agent, 'Q-learning')
    
    # 針對累積獎學繪圖 (平滑處理以提升可讀性)
    window = 10
    sarsa_rewards_smoothed = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
    q_rewards_smoothed = np.convolve(q_learning_rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(10, 6))
    plt.plot(sarsa_rewards_smoothed, label='SARSA', alpha=0.9)
    plt.plot(q_rewards_smoothed, label='Q-learning', alpha=0.9)
    plt.axhline(-13, color='gray', linestyle='--', label='Optimal Path Return') # 最佳路徑是 13 步，獎勵為 -13
    plt.xlabel(f'Episodes (Smoothed over {window} episodes)')
    plt.ylabel('Sum of rewards during episode')
    plt.title('SARSA vs Q-learning on Cliff Walking')
    plt.ylim([-100, 0])  # 設定合理範圍方便觀察差異
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    main()
