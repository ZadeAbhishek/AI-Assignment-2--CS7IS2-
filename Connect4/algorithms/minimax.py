import time
from algorithms.baseline import undo_move  # Assuming undo_move is identical; otherwise, use a common utility.

# Global node counter
node_count = 0

def evaluate_board(game, player):
    """
    A simple evaluation heuristic.
    Currently returns 0. 
    You can enhance this by considering potential wins, two/three in a rows, etc.
    """
    return 0

def minimax_connect4(game, player, depth, alpha=-float('inf'), beta=float('inf'), start_time=None, time_limit=1800):
    global node_count
    node_count += 1

    # Time check: if time limit reached, return evaluation immediately.
    if start_time and (time.time() - start_time) > time_limit:
        return {"position": None, "score": evaluate_board(game, player)}

    max_player = 'O'  # Computer
    other_player = 'X' if player == 'O' else 'O'
    
    # Terminal condition: previous move wins.
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