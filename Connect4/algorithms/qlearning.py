import random
import pickle  # Optional: for saving/loading the model

# Global Q-learning variables.
Q_table = {}       # Maps state (as string) -> dict of {action: Q-value}
state_visits = {}  # Maps state (as string) -> visit count
last_state = None  # The previous state (as a string)
last_action = None # The last action taken
ALPHA = 1          # Learning rate
GAMMA = 0.9        # Discount factor
EPSILON = 0.1      # Exploration rate (lowered to encourage exploitation)
MAX_Q_TABLE_SIZE = 10000  # Maximum number of states to store

def prune_Q_table():
    """Remove a percentage of the least visited states if Q_table grows too large."""
    global Q_table, state_visits
    if len(Q_table) > MAX_Q_TABLE_SIZE:
        num_to_remove = len(Q_table) // 10
        states_sorted = sorted(state_visits.items(), key=lambda x: x[1])
        for i in range(num_to_remove):
            state_to_remove = states_sorted[i][0]
            Q_table.pop(state_to_remove, None)
            state_visits.pop(state_to_remove, None)

def state_str(game):
    """Return a string representation of the board state."""
    return ''.join(''.join(row) for row in game.board)

def evaluate_window(window, player):
    """
    Evaluate a window (list of 4 cells) for the given player.
    Returns a score based on:
      - 4 in a row: high positive score.
      - 3 in a row with an empty cell: moderate positive score.
      - 2 in a row with two empty cells: small positive score.
    Also penalizes opponent threats.
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
    This function scores the board for a given player by:
      - Rewarding center column occupancy.
      - Evaluating all windows (groups of 4 cells) horizontally, vertically, and diagonally.
    """
    score = 0
    board = game.board
    rows = game.rows
    cols = game.cols
    opponent = 'O' if player == 'X' else 'X'

    # Score center column: control of the center is advantageous.
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
    global Q_table, state_visits, last_state, last_action
    current_state = state_str(game)
    available = game.available_moves()
    
    # Increment the visit count for the current state.
    if current_state in state_visits:
        state_visits[current_state] += 1
    else:
        state_visits[current_state] = 1

    # Initialize Q_table for the current state if not already done.
    if current_state not in Q_table:
        Q_table[current_state] = {a: 0.0 for a in available}
    
    # Epsilon-greedy action selection.
    if random.random() < EPSILON:
        action = random.choice(available)
    else:
        action = max(available, key=lambda a: Q_table[current_state].get(a, 0.0))
    
    # Update Q-value for the last move using the evaluation function as the intermediate reward.
    if last_state is not None and last_action is not None:
        if last_state not in Q_table:
            Q_table[last_state] = {}
        old_value = Q_table[last_state].get(last_action, 0.0)
        next_max = max(Q_table[current_state].values()) if Q_table[current_state] else 0.0
        # The intermediate reward is computed by the heuristic evaluation.
        reward = evaluate_board(game, player)
        Q_table[last_state][last_action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
    
    last_state = current_state
    last_action = action
    
    prune_Q_table()
    
    return action

def update_terminal_connect4(last_reward):
    """Update Q_table for the terminal state using the provided terminal reward."""
    global Q_table, last_state, last_action, ALPHA
    if last_state is not None and last_action is not None:
        if last_state not in Q_table:
            Q_table[last_state] = {}
        old_value = Q_table[last_state].get(last_action, 0.0)
        Q_table[last_state][last_action] = old_value + ALPHA * (last_reward - old_value)
    last_state = None
    last_action = None

# Optional: Functions to save and load the Q-learning model.
def save_model(filename="qlearning_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(Q_table, f)
    print(f"Q-learning model saved to {filename}")

def load_model(filename="qlearning_model.pkl"):
    global Q_table
    try:
        with open(filename, "rb") as f:
            Q_table = pickle.load(f)
        print(f"Q-learning model loaded from {filename}")
    except FileNotFoundError:
        print("No saved Q-learning model found. Starting fresh.")