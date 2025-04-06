import time
from algorithms.baseline import undo_move

# Global node counter.
node_count = 0
ALPHA = -float('inf')
BETA = float('inf')

def evaluate_board(game, player):
    """
    A simple evaluation heuristic.
    Currently returns 0.
    You can enhance this by considering potential wins, near-wins, etc.
    """
    return 0

def minimax_connect4(game, player, depth, alpha=-float('inf'), beta=float('inf'), start_time=None, time_limit=1800):
    global node_count
    node_count += 1

    # Check for time limit.
    if start_time and (time.time() - start_time) > time_limit:
        return {"position": None, "score": evaluate_board(game, player)}

    max_player = 'O'  # Computer.
    other_player = 'X' if player == 'O' else 'O'
    
    # Terminal conditions.
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
    global node_count
    node_count += 1

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
    # Track if alpha-beta pruning is used
    use_alpha_beta = True  # As this function is for alpha-beta, it will always use alpha-beta pruning.
    
    result = minimax_connect4(game, player, depth, alpha, beta, start_time, time_limit)
    
    # Return results along with alpha-beta parameters
    return {
        "position": result["position"],
        "score": result["score"],
        "use_alpha_beta": use_alpha_beta,
        "alpha": alpha,
        "beta": beta
    }

def minimax_no_ab_connect4_with_tracking(game, player, depth, start_time=None, time_limit=1800):
    # Track if alpha-beta pruning is not used
    use_alpha_beta = False
    
    result = minimax_no_ab_connect4(game, player, depth, start_time, time_limit)
    
    # Return results along with alpha-beta parameters
    return {
        "position": result["position"],
        "score": result["score"],
        "use_alpha_beta": use_alpha_beta,
        "alpha": None,
        "beta": None
    }