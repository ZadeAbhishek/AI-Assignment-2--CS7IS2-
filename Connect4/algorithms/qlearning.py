# qlearning.py

import random
import pickle
import os
import psutil
import time

# Q-learning state
Q_table = {}
state_visits = {}
last_state = None
last_action = None

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.2
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995


def get_dynamic_memory_threshold(fraction=0.1):
    mem = psutil.virtual_memory()
    return (mem.available / (1024 * 1024)) * fraction


def check_memory_usage():
    process = psutil.Process(os.getpid())
    used = process.memory_info().rss / (1024 * 1024)
    return used > get_dynamic_memory_threshold(0.1)


def save_Q_table_to_disk():
    global Q_table, state_visits
    timestamp = int(time.time())
    filename = f"qlearning_model_{timestamp}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(dict(sorted(Q_table.items(), key=lambda x: state_visits.get(x[0], 0), reverse=True)[:50000]), f)
    print(f"[INFO] Q-table saved to: {filename}")
    Q_table = {}
    state_visits.clear()


def state_str(game):
    return ''.join(''.join(row) for row in game.board)


def evaluate_window(window, player):
    score = 0
    opponent = 'O' if player == 'X' else 'X'

    if window.count(player) == 4:
        score += 1000
    elif window.count(player) == 3 and window.count(' ') == 1:
        score += 50
    elif window.count(player) == 2 and window.count(' ') == 2:
        score += 10

    if window.count(opponent) == 3 and window.count(' ') == 1:
        score += 40  # reward blocking

    return score


def evaluate_board(game, player):
    score = 0
    board = game.board
    rows = game.rows
    cols = game.cols
    center_col = cols // 2
    score += sum([1 for row in board if row[center_col] == player]) * 3

    for r in range(rows):
        for c in range(cols - 3):
            score += evaluate_window(board[r][c:c + 4], player)

    for c in range(cols):
        col_array = [board[r][c] for r in range(rows)]
        for r in range(rows - 3):
            score += evaluate_window(col_array[r:r + 4], player)

    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += evaluate_window(window, player)

    for r in range(3, rows):
        for c in range(cols - 3):
            window = [board[r - i][c + i] for i in range(4)]
            score += evaluate_window(window, player)

    return score


def q_learning_move_connect4(game, player):
    global Q_table, state_visits, last_state, last_action, EPSILON
    current_state = state_str(game)
    available = game.available_moves()

    state_visits[current_state] = state_visits.get(current_state, 0) + 1
    if current_state not in Q_table:
        Q_table[current_state] = {a: 0.0 for a in available}

    if random.random() < EPSILON:
        action = random.choice(available)
    else:
        action = max(available, key=lambda a: Q_table[current_state].get(a, 0.0))

    if last_state and last_action:
        if last_state not in Q_table:
            Q_table[last_state] = {a: 0.0 for a in available}
        old_q = Q_table[last_state].get(last_action, 0.0)
        future_q = max(Q_table[current_state].values()) if Q_table[current_state] else 0.0
        reward = 0.01 * evaluate_board(game, player)
        Q_table[last_state][last_action] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)

    last_state = current_state
    last_action = action

    if check_memory_usage():
        save_Q_table_to_disk()

    return action


def update_terminal_connect4(last_reward):
    global Q_table, last_state, last_action
    if last_state and last_action:
        if last_state not in Q_table:
            Q_table[last_state] = {}
        old_q = Q_table[last_state].get(last_action, 0.0)
        Q_table[last_state][last_action] = old_q + ALPHA * (last_reward - old_q)
    reset_episode_state()


def reset_episode_state():
    global last_state, last_action, EPSILON
    last_state = None
    last_action = None
    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)


def save_model(filename="qlearning_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(Q_table, f)
    print(f"[INFO] Q-learning model saved to {filename}")


def load_model(filename="qlearning_model.pkl"):
    global Q_table
    try:
        with open(filename, "rb") as f:
            Q_table = pickle.load(f)
        print(f"[INFO] Q-learning model loaded from {filename}")
    except FileNotFoundError:
        print("[WARN] No saved Q-table found. Starting fresh.")
