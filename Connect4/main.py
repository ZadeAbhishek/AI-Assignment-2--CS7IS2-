import random
import os
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
from game import Connect4
from algorithms import minimax, qlearning, baseline

def get_computer_move(game, computer_letter, algorithm, use_alpha_beta, depth=4, time_limit=1800):
    if algorithm == '1':
        start_time = time.time()
        if use_alpha_beta:
            # Explicitly pass alpha and beta parameters.
            alpha = -float('inf')
            beta = float('inf')
            move_info = minimax.minimax_connect4(game, computer_letter, depth, alpha, beta, start_time=start_time, time_limit=time_limit)
        else:
            move_info = minimax.minimax_no_ab_connect4(game, computer_letter, depth, start_time=start_time, time_limit=time_limit)
        print(f"Minimax visited {minimax.node_count} nodes.")
        # Reset the counter for next call
        minimax.node_count = 0
        return move_info["position"]
    elif algorithm == '2':
        return qlearning.q_learning_move_connect4(game, computer_letter)
    else:
        return random.choice(game.available_moves())

def play_game_connect4(algorithm, use_alpha_beta, depth=4):
    game = Connect4()
    user_letter = 'X'
    computer_letter = 'O'
    
    print("\nNew Connect 4 game! Baseline (User) vs Computer")
    game.print_board()
    
    turn = 'user' if random.random() < 0.5 else 'computer'
    while game.empty_squares():
        if turn == 'user':
            valid_moves = game.available_moves()
            print("Available moves:", valid_moves)
            move = baseline.baseline_move_connect4(game, user_letter)
            print("\nBaseline (User) chooses move:", move)
            game.make_move(move, user_letter)
            print("\nBoard after User's move:")
            game.print_board()
            if game.current_winner == user_letter:
                print("Baseline (User) wins!")
                if algorithm == '2':
                    # Use -10 penalty for a loss.
                    qlearning.update_terminal_connect4(-10)
                return "user"
            turn = 'computer'
        else:
            print("\nComputer is thinking...")
            move = get_computer_move(game, computer_letter, algorithm, use_alpha_beta, depth)
            print("\nComputer chooses move:", move)
            game.make_move(move, computer_letter)
            print("\nBoard after Computer's move:")
            game.print_board()
            if game.current_winner == computer_letter:
                print("Computer wins!")
                if algorithm == '2':
                    # Use +10 reward for a win.
                    qlearning.update_terminal_connect4(10)
                return "computer"
            turn = 'user'
    print("It's a tie!")
    if algorithm == '2':
        qlearning.update_terminal_connect4(0)
    return "tie"

def save_results(results, parameters, folder_prefix="connect4_results"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{folder_prefix}_{now}"
    os.makedirs(folder_name, exist_ok=True)
    
    csv_file = os.path.join(folder_name, "results.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Game Number", "Result", "User Score", "Computer Score"])
        for row in results:
            writer.writerow(row)
    print(f"CSV results saved to {csv_file}")
    
    games = [row[0] for row in results]
    user_scores = [row[2] for row in results]
    computer_scores = [row[3] for row in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(games, user_scores, label="User (Baseline)")
    plt.plot(games, computer_scores, label="Computer")
    plt.xlabel("Game Number")
    plt.ylabel("Cumulative Score")
    plt.title("Cumulative Score Over Connect 4 Games")
    plt.legend()
    graph_file = os.path.join(folder_name, "cumulative_scores.png")
    plt.savefig(graph_file)
    plt.close()
    print(f"Graph saved to {graph_file}")
    
    params_file = os.path.join(folder_name, "parameters.txt")
    with open(params_file, 'w') as pf:
        pf.write("Parameters used:\n")
        pf.write(f"Algorithm: {'Minimax' if parameters['algorithm']=='1' else 'Q-Learning'}\n")
        if parameters['algorithm'] == '1':
            pf.write(f"Alpha-Beta: {parameters['alpha_beta']}\n")
            pf.write(f"Depth: {parameters['depth']}\n")
        if parameters['algorithm'] == '2':
            pf.write(f"ALPHA: {parameters['ALPHA']}\n")
            pf.write(f"GAMMA: {parameters['GAMMA']}\n")
            pf.write(f"EPSILON: {parameters['EPSILON']}\n")
    print(f"Parameters saved to {params_file}")

def main():
    user_score = 0
    computer_score = 0
    results = []
    
    algorithm = input("Choose an algorithm for the computer (1: Minimax, 2: Tabular Q-Learning): ").strip()
    use_alpha_beta = False
    if algorithm == '1':
        use_alpha_beta = input("Use alpha-beta pruning for minimax? (y/n): ").strip().lower() == 'y'
    depth = 4
    if algorithm == '1':
        depth = int(input("Enter depth limit for minimax search: "))
    
    params = {
        "algorithm": algorithm,
        "alpha_beta": use_alpha_beta if algorithm=='1' else None,
        "depth": depth if algorithm=='1' else None,
        "ALPHA": qlearning.ALPHA if algorithm=='2' else None,
        "GAMMA": qlearning.GAMMA if algorithm=='2' else None,
        "EPSILON": qlearning.EPSILON if algorithm=='2' else None
    }
    
    total_games = int(input("How many games do you want to play? "))
    
    for i in range(total_games):
        print(f"\nStarting Connect 4 game {i+1} of {total_games}...")
        result = play_game_connect4(algorithm, use_alpha_beta, depth)
        if result == "user":
            user_score += 1
        elif result == "computer":
            computer_score += 1
        
        results.append([i+1, result, user_score, computer_score])
        print("\nCurrent Score:")
        print("User (Baseline):", user_score, "Computer:", computer_score)
    
    print("\nFinal Score:")
    print("User (Baseline):", user_score, "Computer:", computer_score)
    save_results(results, params, folder_prefix="connect4_results")
    print("Thanks for playing!")

if __name__ == '__main__':
    main()