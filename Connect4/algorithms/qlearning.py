import random
import pickle  # For saving/loading the model
import os
import psutil  # For checking memory usage
import time

# Global Q-learning variables.
Q_table = {}       # Maps state (as string) -> dict of {action: Q-value}
state_visits = {}  # Maps state (as string) -> visit count
last_state = None  # The previous state (as a string)
last_action = None # The last action taken
ALPHA = 0.01        # Learning rate
GAMMA = 0.999        # Discount factor
EPSILON = 0.001      # Exploration rate

def get_dynamic_memory_threshold(fraction=0.1):
    """
    Returns a dynamic memory threshold (in MB) based on a fraction of available memory.
    For example, fraction=0.1 returns 10% of available memory.
    """
    mem = psutil.virtual_memory()
    available_mb = mem.available / (1024 * 1024)
    return available_mb * fraction

def check_memory_usage():
    """
    Check the current memory usage (in MB) of this process.
    Returns True if usage exceeds the dynamic threshold.
    """
    process = psutil.Process(os.getpid())
    mem_used_mb = process.memory_info().rss / (1024 * 1024)
    threshold_mb = get_dynamic_memory_threshold(0.1)  # 10% of available memory
    # Uncomment next line to debug memory usage:
    # print(f"Memory used: {mem_used_mb:.2f} MB; Threshold: {threshold_mb:.2f} MB")
    return mem_used_mb > threshold_mb

def save_Q_table_to_disk():
    """
    Save the entire Q_table to disk (with a timestamped filename)
    and reset the in-memory Q_table and state_visits.
    """
    global Q_table, state_visits
    filename = f"qlearning_model_{int(time.time())}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(Q_table, f)
    print(f"Q-learning model saved to disk: {filename}")
    # Reset in-memory tables.
    Q_table = {}
    state_visits.clear()

def state_str(game):
    """Return a string representation of the board state (assumes game.board is a 2D list)."""
    return ''.join(''.join(row) for row in game.board)

def evaluate_window(window, player):
    """
    Evaluate a window (list of 4 cells) for the given player.
    
    Scoring:
      - 4 in a row: +100
      - 3 in a row with 1 empty: +5
      - 2 in a row with 2 empties: +2
      - Opponent has 3 in a row with 1 empty: -4
    """
    score = 0
    opponent = 'O' if player == 'X' else 'X'
    
    if window.count(player) == 4:
        score += 100
    elif window.count(player) == 3 and window.count(' ') == 1:
        score += 5
    elif window.count(player) == 2 and window.count(' ') == 2:
        score += 2

    if window.count(opponent) == 3 and window.count(' ') == 1:
        score -= 4

    return score

def evaluate_board(game, player):
    """
    A heuristic evaluation function for Connect4.
    
    Scores the board for a given player by:
      - Rewarding center column occupancy.
      - Summing scores from all windows (4-cell groups) horizontally, vertically, and diagonally.
    """
    score = 0
    board = game.board
    rows = game.rows
    cols = game.cols
    opponent = 'O' if player == 'X' else 'X'
    
    # Center column reward.
    center_col = cols // 2
    center_count = sum([1 for row in board if row[center_col] == player])
    score += center_count * 3

    # Horizontal windows.
    for r in range(rows):
        for c in range(cols - 3):
            window = board[r][c:c+4]
            score += evaluate_window(window, player)
    
    # Vertical windows.
    for c in range(cols):
        col_array = [board[r][c] for r in range(rows)]
        for r in range(rows - 3):
            window = col_array[r:r+4]
            score += evaluate_window(window, player)
    
    # Diagonal (positive slope) windows.
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, player)
    
    # Diagonal (negative slope) windows.
    for r in range(3, rows):
        for c in range(cols - 3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, player)
    
    return score

def q_learning_move_connect4(game, player):
    """
    Select an action using an epsilon-greedy policy.
    Update the Q-value for the previous move using the heuristic evaluation as an intermediate reward.
    """
    global Q_table, state_visits, last_state, last_action
    current_state = state_str(game)
    available = game.available_moves()
    
    # Increment visitation count.
    if current_state in state_visits:
        state_visits[current_state] += 1
    else:
        state_visits[current_state] = 1

    # Initialize Q_table for current state if not present.
    if current_state not in Q_table:
        Q_table[current_state] = {a: 0.0 for a in available}
    
    # Epsilon-greedy action selection.
    if random.random() < EPSILON:
        action = random.choice(available)
    else:
        action = max(available, key=lambda a: Q_table[current_state].get(a, 0.0))
    
    # Update Q-value for the previous move (if available).
    if last_state is not None and last_action is not None:
        if last_state not in Q_table:
            Q_table[last_state] = {a: 0.0 for a in available}
        old_value = Q_table[last_state].get(last_action, 0.0)
        next_max = max(Q_table[current_state].values()) if Q_table[current_state] else 0.0
        reward = evaluate_board(game, player)
        Q_table[last_state][last_action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
    
    last_state = current_state
    last_action = action

    # If memory usage is too high, save the Q_table to disk and reset it.
    if check_memory_usage():
        save_Q_table_to_disk()

    return action

def update_terminal_connect4(last_reward):
    """
    Update the Q-table for the terminal state with the given reward.
    Then reset last_state and last_action.
    """
    global Q_table, last_state, last_action, ALPHA
    if last_state is not None and last_action is not None:
        if last_state not in Q_table:
            Q_table[last_state] = {}
        old_value = Q_table[last_state].get(last_action, 0.0)
        Q_table[last_state][last_action] = old_value + ALPHA * (last_reward - old_value)
    last_state = None
    last_action = None

def save_model(filename="qlearning_model.pkl"):
    """Manually save the Q_table to disk."""
    with open(filename, "wb") as f:
        pickle.dump(Q_table, f)
    print(f"Q-learning model saved to {filename}")

def load_model(filename="qlearning_model.pkl"):
    """Manually load the Q_table from disk."""
    global Q_table
    try:
        with open(filename, "rb") as f:
            Q_table = pickle.load(f)
        print(f"Q-learning model loaded from {filename}")
    except FileNotFoundError:
        print("No saved Q-learning model found. Starting fresh.")