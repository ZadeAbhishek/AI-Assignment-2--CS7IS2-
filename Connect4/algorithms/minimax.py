import time
from algorithms.baseline import undo_move

node_count = 0
states_explored = 0
ALPHA = -float('inf')
BETA = float('inf')

def evaluate_board(game, player):
    opponent = 'X' if player == 'O' else 'O'
    score = 0
    for row in range(game.rows):
        for col in range(game.columns):
            if game.board[row][col] == player:
                score += evaluate_direction(game, row, col, 1, 0, player)
                score += evaluate_direction(game, row, col, 0, 1, player)
                score += evaluate_direction(game, row, col, 1, 1, player)
                score += evaluate_direction(game, row, col, 1, -1, player)
            elif game.board[row][col] == opponent:
                score -= evaluate_direction(game, row, col, 1, 0, opponent)
                score -= evaluate_direction(game, row, col, 0, 1, opponent)
                score -= evaluate_direction(game, row, col, 1, 1, opponent)
                score -= evaluate_direction(game, row, col, 1, -1, opponent)
    return score

def evaluate_direction(game, row, col, d_row, d_col, player):
    count = 0
    for i in range(4):
        r, c = row + i * d_row, col + i * d_col
        if 0 <= r < game.rows and 0 <= c < game.columns:
            if game.board[r][c] == player:
                count += 1
            elif game.board[r][c] != ' ':
                return 0
        else:
            return 0
    if count == 4:
        return 100
    elif count == 3:
        return 10
    elif count == 2:
        return 1
    else:
        return 0

def minimax_connect4(game, player, depth, alpha=-float('inf'), beta=float('inf'), start_time=None, time_limit=1800):
    global node_count, states_explored
    node_count += 1
    states_explored += 1
    if start_time and (time.time() - start_time) > time_limit:
        return {"position": None, "score": evaluate_board(game, player)}
    max_player = 'O'
    other_player = 'X' if player == 'O' else 'O'
    if game.current_winner == other_player:
        return {"position": None, "score": (len(game.available_moves()) + 1) if other_player == max_player else -1 * (len(game.available_moves()) + 1)}
    elif depth == 0 or not game.empty_squares():
        return {"position": None, "score": evaluate_board(game, player)}
    if player == max_player:
        best = {"position": None, "score": -float('inf')}
    else:
        best = {"position": None, "score": float('inf')}
    for move in game.available_moves():
        game.make_move(move, player)
        sim_score = minimax_connect4(game, other_player, depth - 1, alpha, beta, start_time, time_limit)
        undo_move(game, move)
        sim_score["position"] = move
        if player == max_player:
            if sim_score["score"] > best["score"]:
                best = sim_score
            alpha = max(alpha, best["score"])
        else:
            if sim_score["score"] < best["score"]:
                best = sim_score
            beta = min(beta, best["score"])
        if beta <= alpha:
            break
    return best

def minimax_no_ab_connect4(game, player, depth, start_time=None, time_limit=1800):
    global node_count, states_explored
    node_count += 1
    states_explored += 1
    if start_time and (time.time() - start_time) > time_limit:
        return {"position": None, "score": evaluate_board(game, player)}
    max_player = 'O'
    other_player = 'X' if player == 'O' else 'O'
    if game.current_winner == other_player:
        return {"position": None, "score": (len(game.available_moves()) + 1) if other_player == max_player else -1 * (len(game.available_moves()) + 1)}
    elif depth == 0 or not game.empty_squares():
        return {"position": None, "score": evaluate_board(game, player)}
    if player == max_player:
        best = {"position": None, "score": -float('inf')}
    else:
        best = {"position": None, "score": float('inf')}
    for move in game.available_moves():
        game.make_move(move, player)
        sim_score = minimax_no_ab_connect4(game, other_player, depth - 1, start_time, time_limit)
        undo_move(game, move)
        sim_score["position"] = move
        if player == max_player:
            if sim_score["score"] > best["score"]:
                best = sim_score
        else:
            if sim_score["score"] < best["score"]:
                best = sim_score
    return best

def minimax_connect4_with_tracking(game, player, depth, alpha=-float('inf'), beta=float('inf'), start_time=None, time_limit=1800):
    use_alpha_beta = True
    result = minimax_connect4(game, player, depth, alpha, beta, start_time, time_limit)
    return {
        "position": result["position"],
        "score": result["score"],
        "use_alpha_beta": use_alpha_beta,
        "alpha": alpha,
        "beta": beta
    }

def minimax_no_ab_connect4_with_tracking(game, player, depth, start_time=None, time_limit=1800):
    use_alpha_beta = False
    result = minimax_no_ab_connect4(game, player, depth, start_time, time_limit)
    return {
        "position": result["position"],
        "score": result["score"],
        "use_alpha_beta": use_alpha_beta,
        "alpha": None,
        "beta": None
    }

def get_states_explored():
    return states_explored