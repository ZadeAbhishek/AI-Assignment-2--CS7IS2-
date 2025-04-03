def minimax(game, player, alpha=-float('inf'), beta=float('inf')):
    max_player = 'O'  # Computer is maximizing
    other_player = 'X' if player == 'O' else 'O'
    
    # Terminal condition: if previous move won.
    if game.current_winner == other_player:
        return {"position": None, "score": (len(game.available_moves()) + 1) if other_player == max_player else -1 * (len(game.available_moves()) + 1)}
    elif not game.empty_squares():
        return {"position": None, "score": 0}
    
    if player == max_player:
        best = {"position": None, "score": -float('inf')}
    else:
        best = {"position": None, "score": float('inf')}
    
    for possible_move in game.available_moves():
        game.make_move(possible_move, player)
        sim_score = minimax(game, other_player, alpha, beta)
        # Undo move.
        game.board[possible_move] = ' '
        game.current_winner = None
        sim_score["position"] = possible_move

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

def minimax_no_ab(game, player):
    max_player = 'O'
    other_player = 'X' if player == 'O' else 'O'
    
    if game.current_winner == other_player:
        return {"position": None, "score": (len(game.available_moves()) + 1) if other_player == max_player else -1 * (len(game.available_moves()) + 1)}
    elif not game.empty_squares():
        return {"position": None, "score": 0}

    if player == max_player:
        best = {"position": None, "score": -float('inf')}
    else:
        best = {"position": None, "score": float('inf')}
    
    for possible_move in game.available_moves():
        game.make_move(possible_move, player)
        sim_score = minimax_no_ab(game, other_player)
        game.board[possible_move] = ' '
        game.current_winner = None
        sim_score["position"] = possible_move

        if player == max_player:
            if sim_score["score"] > best["score"]:
                best = sim_score
        else:
            if sim_score["score"] < best["score"]:
                best = sim_score

    return best