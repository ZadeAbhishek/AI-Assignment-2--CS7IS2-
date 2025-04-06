import random
import pickle
import time

Q_table = {}
last_state = None
last_action = None

ALPHA = 0.3
GAMMA = 0.9
EPSILON = 0.7
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.999
SAVE_FREQUENCY = 1000

game_counter = 0
last_save_time = time.time()

def state_str(game, player):
    return ''.join(game.board) + ":" + player

def save_Q_table_to_disk():
    global Q_table
    with open("qlearning_model.pkl", "wb") as f:
        pickle.dump(Q_table, f)
    print("[INFO] Q-table saved to qlearning_model.pkl")

def q_learning_move(game, player):
    global Q_table, last_state, last_action, EPSILON, game_counter

    current_state = state_str(game, player)
    available_moves = game.available_moves()

    if current_state not in Q_table:
        Q_table[current_state] = {move: 0.0 for move in available_moves}

    if random.random() < EPSILON:
        center = 4
        if center in available_moves and random.random() < 0.7:
            action = center
        else:
            action = random.choice(available_moves)
    else:
        action = max(available_moves, key=lambda a: Q_table[current_state].get(a, 0.0))

    if last_state is not None and last_action is not None:
        future_q = max(Q_table[current_state].values()) if Q_table[current_state] else 0.0
        old_q = Q_table[last_state].get(last_action, 0.0)
        reward = 0
        Q_table[last_state][last_action] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)

    last_state = current_state
    last_action = action
    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

    game_counter += 1
    if game_counter % SAVE_FREQUENCY == 0:
        save_Q_table_to_disk()

    return action

def update_terminal(reward):
    global Q_table, last_state, last_action
    if last_state is not None and last_action is not None:
        old_q = Q_table[last_state].get(last_action, 0.0)
        Q_table[last_state][last_action] = old_q + ALPHA * (reward - old_q)
    reset_episode()

def reset_episode():
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