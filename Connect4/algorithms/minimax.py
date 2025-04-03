def undo_move(game, col):
    """Undo the topmost move in the given column."""
    for row in range(game.rows):
        if game.board[row][col] != ' ':
            game.board[row][col] = ' '
            game.current_winner = None
            break

def minimax_connect4(game, player, depth, alpha=-float('inf'), beta=float('inf')):
    max_player = 'O'  # Computer
    other_player = 'X' if player == 'O' else 'O'
    
    # Terminal: if last move won.
    if game.current_winner == other_player:
        return {"position": None, "score": (len(game.available_moves()) + 1) if other_player == max_player else -1 * (len(game.available_moves()) + 1)}
    elif depth == 0 or not game.empty_squares():
        # Simple evaluation (tie state).
        return {"position": None, "score": 0}
    
    if player == max_player:
        best = {"position": None, "score": -float('inf')}
    else:
        best = {"position": None, "score": float('inf')}
    
    for move in game.available_moves():
        game.make_move(move, player)
        sim_score = minimax_connect4(game, other_player, depth - 1, alpha, beta)
        # Undo move.
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

def minimax_no_ab_connect4(game, player, depth):
    max_player = 'O'
    other_player = 'X' if player == 'O' else 'O'
    
    if game.current_winner == other_player:
        return {"position": None, "score": (len(game.available_moves()) + 1) if other_player == max_player else -1 * (len(game.available_moves()) + 1)}
    elif depth == 0 or not game.empty_squares():
        return {"position": None, "score": 0}
    
    if player == max_player:
        best = {"position": None, "score": -float('inf')}
    else:
        best = {"position": None, "score": float('inf')}
    
    for move in game.available_moves():
        game.make_move(move, player)
        sim_score = minimax_no_ab_connect4(game, other_player, depth - 1)
        undo_move(game, move)
        sim_score["position"] = move
        
        if player == max_player:
            if sim_score["score"] > best["score"]:
                best = sim_score
        else:
            if sim_score["score"] < best["score"]:
                best = sim_score
    return best