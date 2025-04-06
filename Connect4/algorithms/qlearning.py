import random
import pickle
import os
import time
import numpy as np

Q_table = {}
state_visits = {}
last_state = None
last_action = None

ALPHA = 0.3
GAMMA = 0.9
EPSILON = 0.7
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9999

SAVE_FREQUENCY = 5000
game_counter = 0
last_save_time = time.time()

def state_str(game, player):
    board_str = ''.join(''.join(row) for row in game.board)
    return f"{board_str}:{player}"

def evaluate_window(window, player):
    opponent = 'O' if player == 'X' else 'X'
    score = 0
    if window.count(player) == 4:
        return 1000
    if window.count(opponent) == 3 and window.count(' ') == 1:
        return 50
    if window.count(player) == 3 and window.count(' ') == 1:
        score += 20
    elif window.count(player) == 2 and window.count(' ') == 2:
        score += 5
    if window.count(opponent) == 2 and window.count(' ') == 2:
        score += 3
    return score

def evaluate_board(game, player):
    score = 0
    board = game.board
    rows = game.rows
    cols = game.cols
    center_col = cols // 2
    center_array = [board[r][center_col] for r in range(rows)]
    center_count = center_array.count(player)
    score += center_count * 10
    for r in range(rows):
        for c in range(cols - 3):
            window = [board[r][c+i] for i in range(4)]
            score += evaluate_window(window, player)
    for c in range(cols):
        for r in range(rows - 3):
            window = [board[r+i][c] for i in range(4)]
            score += evaluate_window(window, player)
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, player)
    for r in range(3, rows):
        for c in range(cols - 3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, player)
    return score

def save_Q_table_to_disk(force=False):
    global Q_table, game_counter, last_save_time
    game_counter += 1
    current_time = time.time()
    if force or (game_counter % SAVE_FREQUENCY == 0 and current_time - last_save_time > 60):
        filename = f"qlearning_model.pkl"
        with open(filename, "wb") as f:
            pickle.dump(Q_table, f)
        print(f"[INFO] Q-table saved to: {filename}")
        last_save_time = current_time
        if game_counter % (SAVE_FREQUENCY * 10) == 0:
            timestamp = int(time.time())
            backup_filename = f"qlearning_backup_{timestamp}.pkl"
            with open(backup_filename, "wb") as f:
                pickle.dump(Q_table, f)

def q_learning_move_connect4(game, player):
    global Q_table, state_visits, last_state, last_action, EPSILON
    current_state = state_str(game, player)
    available_moves = game.available_moves()
    if current_state not in Q_table:
        Q_table[current_state] = {move: 0.0 for move in available_moves}
    state_visits[current_state] = state_visits.get(current_state, 0) + 1
    for move in available_moves:
        game_copy = Connect4()
        game_copy.board = [row[:] for row in game.board]
        game_copy.make_move(move, player)
        if game_copy.current_winner == player:
            if current_state not in Q_table:
                Q_table[current_state] = {}
            Q_table[current_state][move] = 100.0
            last_state = current_state
            last_action = move
            return move
    opponent = 'O' if player == 'X' else 'X'
    for move in available_moves:
        game_copy = Connect4()
        game_copy.board = [row[:] for row in game.board]
        game_copy.make_move(move, opponent)
        if game_copy.current_winner == opponent:
            if current_state not in Q_table:
                Q_table[current_state] = {}
            Q_table[current_state][move] = 80.0
            last_state = current_state
            last_action = move
            return move
    if random.random() < EPSILON:
        center_col = game.cols // 2
        if center_col in available_moves and random.random() < 0.7:
            action = center_col
        else:
            action = random.choice(available_moves)
    else:
        if current_state in Q_table and Q_table[current_state]:
            action = max(available_moves, key=lambda a: Q_table[current_state].get(a, 0.0))
        else:
            center_col = game.cols // 2
            if center_col in available_moves:
                action = center_col
            else:
                action = random.choice(available_moves)
    if last_state and last_action:
        if last_state not in Q_table:
            Q_table[last_state] = {}
        reward = evaluate_board(game, player) / 50.0
        future_q = max(Q_table[current_state].values()) if Q_table[current_state] else 0.0
        old_q = Q_table[last_state].get(last_action, 0.0)
        Q_table[last_state][last_action] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)
    last_state = current_state
    last_action = action
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    return action

def update_terminal_connect4(last_reward):
    global Q_table, last_state, last_action
    if last_state and last_action:
        if last_state not in Q_table:
            Q_table[last_state] = {}
        old_q = Q_table[last_state].get(last_action, 0.0)
        Q_table[last_state][last_action] = old_q + ALPHA * (last_reward - old_q)
        save_Q_table_to_disk(force=abs(last_reward) > 5)
    reset_episode_state()

def reset_episode_state():
    global last_state, last_action
    last_state = None
    last_action = None

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

try:
    from game import Connect4
except ImportError:
    class Connect4:
        def __init__(self, rows=6, cols=7):
            self.rows = rows
            self.cols = cols
            self.board = [[' ' for _ in range(cols)] for _ in range(rows)]
            self.current_winner = None