import random

# Global Q-learning variables.
Q_table = {}       # Maps state (as string) -> dict of {action: Q-value}
state_visits = {}  # Maps state (as string) -> visit count
last_state = None  # The previous state as a string
last_action = None # The last action taken
ALPHA = 1          # Learning rate
GAMMA = 0.9        # Discount factor
EPSILON = 1        # Exploration rate
MAX_Q_TABLE_SIZE = 10000  # Maximum states to store

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

def evaluate_board(game, player):
    """
    A simple evaluation function:
    Returns the difference between the count of player's pieces and opponent's pieces.
    """
    player_count = sum(row.count(player) for row in game.board)
    opponent = 'O' if player == 'X' else 'X'
    opponent_count = sum(row.count(opponent) for row in game.board)
    return player_count - opponent_count

def q_learning_move_connect4(game, player):
    global Q_table, state_visits, last_state, last_action
    current_state = state_str(game)
    available = game.available_moves()
    
    # Increment visit count.
    if current_state in state_visits:
        state_visits[current_state] += 1
    else:
        state_visits[current_state] = 1

    # Initialize Q_table for current state if needed.
    if current_state not in Q_table:
        Q_table[current_state] = {a: 0.0 for a in available}
    
    # Epsilon-greedy selection.
    if random.random() < EPSILON:
        action = random.choice(available)
    else:
        action = max(available, key=lambda a: Q_table[current_state].get(a, 0.0))
    
    # Update Q-value for last move.
    if last_state is not None and last_action is not None:
        if last_state not in Q_table:
            Q_table[last_state] = {}
        old_value = Q_table[last_state].get(last_action, 0.0)
        next_max = max(Q_table[current_state].values()) if Q_table[current_state] else 0.0
        # Use evaluation as intermediate reward.
        reward = evaluate_board(game, player)
        Q_table[last_state][last_action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
    
    last_state = current_state
    last_action = action
    
    prune_Q_table()
    
    return action

def update_terminal_connect4(last_reward):
    """Update Q_table for the terminal state using last_reward."""
    global Q_table, last_state, last_action, ALPHA
    if last_state is not None and last_action is not None:
        if last_state not in Q_table:
            Q_table[last_state] = {}
        old_value = Q_table[last_state].get(last_action, 0.0)
        Q_table[last_state][last_action] = old_value + ALPHA * (last_reward - old_value)
    last_state = None
    last_action = None