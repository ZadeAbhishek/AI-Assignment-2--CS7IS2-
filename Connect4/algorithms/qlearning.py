import random

# Global Q-learning variables.
Q_table = {}       # Maps state (as string) -> dict of {action: Q-value}
last_state = None  # The last state encountered by the Q-learning agent
last_action = None # The action taken in the last state
ALPHA = 0.5        # Learning rate
GAMMA = 0.9        # Discount factor
EPSILON = 0.2      # Exploration rate

def state_str(game):
    """Return a string representation of the board state."""
    return ''.join(''.join(row) for row in game.board)

def q_learning_move_connect4(game, player):
    global Q_table, last_state, last_action
    s = state_str(game)
    available = game.available_moves()
    if s not in Q_table:
        Q_table[s] = {a: 0 for a in available}
    if random.random() < EPSILON:
        action = random.choice(available)
    else:
        action = max(available, key=lambda a: Q_table[s].get(a, 0))
    if last_state is not None:
        if last_state not in Q_table:
            Q_table[last_state] = {}
        old_value = Q_table[last_state].get(last_action, 0)
        next_max = max(Q_table[s].values()) if Q_table[s] else 0
        Q_table[last_state][last_action] = old_value + ALPHA * (0 + GAMMA * next_max - old_value)
    last_state = s
    last_action = action
    return action

def update_terminal_connect4(last_reward):
    global Q_table, last_state, last_action, ALPHA
    if last_state is not None:
        old_value = Q_table[last_state].get(last_action, 0)
        Q_table[last_state][last_action] = old_value + ALPHA * (last_reward - old_value)