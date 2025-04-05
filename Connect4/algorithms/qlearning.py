# Optimized qlearning.py with reduced file generation and better performance

import random
import pickle
import os
import time
import numpy as np

# Q-learning state
Q_table = {}
state_visits = {}
last_state = None
last_action = None

# Hyperparameters - significantly adjusted
ALPHA = 0.3       # Increased learning rate for faster updates
GAMMA = 0.9       # Slightly lower discount to focus more on immediate rewards
EPSILON = 0.7     # Much higher initial exploration
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9999

# Control variables
SAVE_FREQUENCY = 5000  # Only save every 50 games
game_counter = 0
last_save_time = time.time()

def state_str(game, player):
    """Create a state representation that includes both board state and player"""
    board_str = ''.join(''.join(row) for row in game.board)
    return f"{board_str}:{player}"

def evaluate_window(window, player):
    """Evaluate a window of 4 positions"""
    opponent = 'O' if player == 'X' else 'X'
    score = 0
    
    # Check for immediate wins (highest priority)
    if window.count(player) == 4:
        return 1000
    
    # Check for blocking opponent's wins (high priority)
    if window.count(opponent) == 3 and window.count(' ') == 1:
        return 50
    
    # My potential wins
    if window.count(player) == 3 and window.count(' ') == 1:
        score += 20
    elif window.count(player) == 2 and window.count(' ') == 2:
        score += 5
    
    # Block opponent's potential
    if window.count(opponent) == 2 and window.count(' ') == 2:
        score += 3
    
    return score

def evaluate_board(game, player):
    """Evaluate current board position for the given player"""
    score = 0
    board = game.board
    rows = game.rows
    cols = game.cols
    
    # Prefer center column - crucial in Connect4
    center_col = cols // 2
    center_array = [board[r][center_col] for r in range(rows)]
    center_count = center_array.count(player)
    score += center_count * 10
    
    # Check all possible winning windows
    
    # Horizontal windows
    for r in range(rows):
        for c in range(cols - 3):
            window = [board[r][c+i] for i in range(4)]
            score += evaluate_window(window, player)
    
    # Vertical windows
    for c in range(cols):
        for r in range(rows - 3):
            window = [board[r+i][c] for i in range(4)]
            score += evaluate_window(window, player)
    
    # Diagonal windows (positive slope)
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, player)
    
    # Diagonal windows (negative slope)
    for r in range(3, rows):
        for c in range(cols - 3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, player)
    
    return score

def save_Q_table_to_disk(force=False):
    """Save Q-table with controlled frequency to prevent file explosion"""
    global Q_table, game_counter, last_save_time
    game_counter += 1
    
    # Only save periodically or if forced
    current_time = time.time()
    if force or (game_counter % SAVE_FREQUENCY == 0 and current_time - last_save_time > 60):
        filename = f"qlearning_model.pkl"
        with open(filename, "wb") as f:
            pickle.dump(Q_table, f)
        print(f"[INFO] Q-table saved to: {filename}")
        last_save_time = current_time
        
        # Optional: create a timestamp backup periodically
        if game_counter % (SAVE_FREQUENCY * 10) == 0:
            timestamp = int(time.time())
            backup_filename = f"qlearning_backup_{timestamp}.pkl"
            with open(backup_filename, "wb") as f:
                pickle.dump(Q_table, f)

def q_learning_move_connect4(game, player):
    """Get the next move using Q-learning"""
    global Q_table, state_visits, last_state, last_action, EPSILON
    
    # Create state representation
    current_state = state_str(game, player)
    available_moves = game.available_moves()
    
    # Initialize state if needed
    if current_state not in Q_table:
        Q_table[current_state] = {move: 0.0 for move in available_moves}
    
    # Track state visits for learning analysis
    state_visits[current_state] = state_visits.get(current_state, 0) + 1
    
    # Use a winning move immediately if available
    for move in available_moves:
        game_copy = Connect4()
        game_copy.board = [row[:] for row in game.board]  # Copy board
        game_copy.make_move(move, player)
        if game_copy.current_winner == player:
            # Learn from this immediate win
            if current_state not in Q_table:
                Q_table[current_state] = {}
            Q_table[current_state][move] = 100.0  # Very high value for winning move
            last_state = current_state
            last_action = move
            return move
    
    # Block opponent's winning move
    opponent = 'O' if player == 'X' else 'X'
    for move in available_moves:
        game_copy = Connect4()
        game_copy.board = [row[:] for row in game.board]  # Copy board
        game_copy.make_move(move, opponent)
        if game_copy.current_winner == opponent:
            # Learn from this important blocking move
            if current_state not in Q_table:
                Q_table[current_state] = {}
            Q_table[current_state][move] = 80.0  # High value for blocking move
            last_state = current_state
            last_action = move
            return move
    
    # Exploration vs exploitation
    if random.random() < EPSILON:
        # Exploration: choose random move, biased toward center
        center_col = game.cols // 2
        if center_col in available_moves and random.random() < 0.7:  # 70% chance to pick center when available
            action = center_col
        else:
            action = random.choice(available_moves)
    else:
        # Exploitation: choose best known move
        if current_state in Q_table and Q_table[current_state]:
            # Find best move based on Q-values
            action = max(available_moves, key=lambda a: Q_table[current_state].get(a, 0.0))
        else:
            # If no Q-value info, prefer center column when available
            center_col = game.cols // 2
            if center_col in available_moves:
                action = center_col
            else:
                action = random.choice(available_moves)
    
    # Learn from previous state-action pair
    if last_state and last_action:
        if last_state not in Q_table:
            Q_table[last_state] = {}
        
        # Calculate reward from board evaluation
        reward = evaluate_board(game, player) / 50.0  # Scale the evaluation
        
        # Get max future Q-value for the update
        future_q = max(Q_table[current_state].values()) if Q_table[current_state] else 0.0
        
        # Q-learning update rule
        old_q = Q_table[last_state].get(last_action, 0.0)
        Q_table[last_state][last_action] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)
    
    # Update state
    last_state = current_state
    last_action = action
    
    # Decay epsilon
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    
    return action

def update_terminal_connect4(last_reward):
    """Update Q-values for terminal states"""
    global Q_table, last_state, last_action
    
    if last_state and last_action:
        if last_state not in Q_table:
            Q_table[last_state] = {}
        
        # Update Q-value with terminal reward
        old_q = Q_table[last_state].get(last_action, 0.0)
        Q_table[last_state][last_action] = old_q + ALPHA * (last_reward - old_q)
        
        # Save the model after significant learning events (win/loss)
        save_Q_table_to_disk(force=abs(last_reward) > 5)
    
    reset_episode_state()

def reset_episode_state():
    """Reset episode state between games"""
    global last_state, last_action
    last_state = None
    last_action = None

def save_model(filename="qlearning_model.pkl"):
    """Save the model explicitly"""
    with open(filename, "wb") as f:
        pickle.dump(Q_table, f)
    print(f"[INFO] Q-learning model saved to {filename}")

def load_model(filename="qlearning_model.pkl"):
    """Load a previously saved model"""
    global Q_table
    try:
        with open(filename, "rb") as f:
            Q_table = pickle.load(f)
        print(f"[INFO] Q-learning model loaded from {filename}")
    except FileNotFoundError:
        print("[WARN] No saved Q-table found. Starting fresh.")

# Ensure Connect4 class is accessible for move validation
try:
    from game import Connect4
except ImportError:
    # Stub implementation for when imported directly
    class Connect4:
        def __init__(self, rows=6, cols=7):
            self.rows = rows
            self.cols = cols
            self.board = [[' ' for _ in range(cols)] for _ in range(rows)]
            self.current_winner = None