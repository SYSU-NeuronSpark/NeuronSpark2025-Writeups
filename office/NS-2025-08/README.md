# NS-2025-08 题解 —— 我搭的靶场

# 赛题背景

路径规划是自动驾驶、机器人导航与智能体控制等领域的核心问题之一。在实际应用中，智能体常常需要在已知或未知的地图上，规避障碍物，自主寻找一条从起点到目标点的最优路径。

本赛题以“逆向设计”的形式展开，即不要求参赛者设计智能体，而是设计用于`评估智能体能力`的挑战性环境。这种模式广泛应用于智能体鲁棒性测试、安全性验证及对抗性训练等任务中。

# 解题思路

本题希望选手结合已知环境等参数信息，基于智能体的缺陷设计路线。关键在于理解智能体的行为逻辑。

## 方向偏好缺陷
```python
self.direction_preference = {
    (1, 0): 1.2,   # 向右
    (0, 1): 1.2,   # 向上
    (-1, 0): 1.0,  # 向左
    (0, -1): 1.0,  # 向下
    (1, 1): 0.6,   # 对角线
    (-1, 1): 0.6,
    (1, -1): 0.6,
    (-1, -1): 0.6
```

智能体对向右和向上的动作有明显偏好（1.2），而对角线移动的偏好值较低（0.6）。这意味着在选择路径时，即使对角线移动可能是更好的选择，智能体也倾向于选择水平或垂直移动。

## 视野范围限制
```python
self.perception_range = 15.0

def filter_obstacles(self, pos, obstacles):
    visible_obstacles = []
    for obs in obstacles:
        dist = math.sqrt((obs["x"] - pos[0])**2 + (obs["y"] - pos[1])**2)
        if dist <= self.perception_range:
            visible_obstacles.append(obs)
    return visible_obstacles
```
智能体只能感知15米范围内的障碍物，这意味着它无法提前规划远处的路径。可以利用这一点，在视野范围外设置障碍物陷阱。

## 避障策略缺陷
```python
def calculate_reward(self, pos, goal, obstacles, prev_pos=None):
    # ...
    obstacle_penalty = 0
    for obs in obstacles:
        dist = math.sqrt((obs["x"] - pos[0])**2 + (obs["y"] - pos[1])**2)
        if dist < self.narrow_passage_threshold:
            obstacle_penalty -= 5.0 / (dist + 0.1)
```
当遇到障碍物时，智能体会在距离小于3米（narrow_passage_threshold）时才开始计算惩罚，且惩罚值随距离增加而迅速减小。这使得智能体容易在复杂的障碍物群中陷入局部最优解。

## 滞留检测缺陷
```python
def choose_action(self, state, valid_actions, current_pos):
    # ...
    pos_tuple = (current_pos[0], current_pos[1])
    is_stuck = self.position_history.count(pos_tuple) > 2
    
    if is_stuck and random.random() < 0.5:
        return random.choice(valid_actions)
```
当智能体在同一位置停留超过2次时，只有50%的概率会随机选择新的动作。这个机制可能不足以让智能体摆脱复杂的困境。

## 有限探索策略
```python
self.exploration_rate = 0.15
```
15%的探索率相对较低，这使得智能体容易陷入已知的次优路径，难以发现更好的解决方案。

## 总结
基于这些缺陷，可以从以下角度设计挑战环境：

1. 在起点和终点之间设置需要对角线移动才能高效通过的障碍物布局
2. 在智能体视野范围外设置障碍物陷阱
3. 创建需要绕远路才能到达的环境，利用智能体对短路径的偏好
4. 设计复杂的迷宫型障碍物布局，迫使智能体在局部区域反复尝试
5. 构造需要通过狭窄通道的路径，利用其避障策略的缺陷