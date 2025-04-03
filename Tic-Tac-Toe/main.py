import random
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

from game import TicTacToe
from algorithms import minimax, qlearning, baseline

def select_algorithm():
    print("Choose an algorithm for the computer:")
    print("1. Minimax")
    print("2. Tabular Q-Learning")
    choice = input("Enter your choice (1/2): ")
    return choice.strip()

def select_alpha_beta():
    response = input("Use alpha-beta pruning for minimax? (y/n): ").strip().lower()
    return response == 'y'

def get_computer_move(game, computer_letter, algorithm, use_alpha_beta):
    if algorithm == '1':
        if use_alpha_beta:
            move_info = minimax.minimax(game, computer_letter)
            return move_info["position"]
        else:
            move_info = minimax.minimax_no_ab(game, computer_letter)
            return move_info["position"]
    elif algorithm == '2':
        return qlearning.q_learning_move(game, computer_letter)
    else:
        return random.choice(game.available_moves())

def play_game(algorithm, use_alpha_beta):
    # Start a new game.
    game = TicTacToe()
    # In this simulation, the baseline strategy is used for the "user"
    # and the chosen algorithm (Minimax or Q-Learning) for the "computer."
    user_letter = 'X'
    computer_letter = 'O'

    print("\nNew game! Baseline (User) vs Computer")
    game.print_board()

    # Randomly choose who goes first.
    turn = 'user' if random.random() < 0.5 else 'computer'

    while game.empty_squares():
        if turn == 'user':
            valid_moves = game.available_moves()
            print("Available moves:", valid_moves)
            move = baseline.baseline_move(game, user_letter)
            print("\nBaseline (User) chooses move:", move)
            game.make_move(move, user_letter)
            print("\nBoard after User's move:")
            game.print_board()
            if game.current_winner == user_letter:
                print("Baseline (User) wins!")
                if algorithm == '2':
                    qlearning.update_terminal(-1)
                return "user"
            turn = 'computer'
        else:
            print("\nComputer is thinking...")
            move = get_computer_move(game, computer_letter, algorithm, use_alpha_beta)
            print("\nComputer chooses move:", move)
            game.make_move(move, computer_letter)
            print("\nBoard after Computer's move:")
            game.print_board()
            if game.current_winner == computer_letter:
                print("Computer wins!")
                if algorithm == '2':
                    qlearning.update_terminal(1)
                return "computer"
            turn = 'user'
    print("It's a tie!")
    if algorithm == '2':
        qlearning.update_terminal(0)
    return "tie"

def save_results(results, parameters):
    # Create a folder with current date and time.
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"results_{now}"
    os.makedirs(folder_name, exist_ok=True)

    # Write CSV file.
    csv_file = os.path.join(folder_name, "results.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Game Number", "Result", "User Score", "Computer Score"])
        for row in results:
            writer.writerow(row)
    print(f"CSV results saved to {csv_file}")

    # Plot graph of cumulative scores.
    games = [row[0] for row in results]
    user_scores = [row[2] for row in results]
    computer_scores = [row[3] for row in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(games, user_scores, label="User (Baseline)")
    plt.plot(games, computer_scores, label="Computer")
    plt.xlabel("Game Number")
    plt.ylabel("Cumulative Score")
    plt.title("Cumulative Score Over Games")
    plt.legend()
    graph_file = os.path.join(folder_name, "cumulative_scores.png")
    plt.savefig(graph_file)
    plt.close()
    print(f"Graph saved to {graph_file}")

    # Save parameters in a text file.
    params_file = os.path.join(folder_name, "parameters.txt")
    with open(params_file, 'w') as pf:
        pf.write("Parameters used:\n")
        pf.write(f"Algorithm: {'Minimax' if parameters['algorithm']=='1' else 'Q-Learning'}\n")
        if parameters['algorithm'] == '1':
            pf.write(f"Alpha-Beta: {parameters['alpha_beta']}\n")
        if parameters['algorithm'] == '2':
            pf.write(f"ALPHA: {parameters['ALPHA']}\n")
            pf.write(f"GAMMA: {parameters['GAMMA']}\n")
            pf.write(f"EPSILON: {parameters['EPSILON']}\n")
    print(f"Parameters saved to {params_file}")

def main():
    user_score = 0
    computer_score = 0
    results = []  # Will store (game_number, result, user_score, computer_score)

    algorithm = select_algorithm()
    use_alpha_beta = False
    if algorithm == '1':
        use_alpha_beta = select_alpha_beta()

    # Gather parameters (for analysis)
    params = {
        "algorithm": algorithm,
        "alpha_beta": use_alpha_beta,
        "ALPHA": qlearning.ALPHA if algorithm=='2' else None,
        "GAMMA": qlearning.GAMMA if algorithm=='2' else None,
        "EPSILON": qlearning.EPSILON if algorithm=='2' else None
    }

    total_games = int(input("How many games do you want to play? "))

    for i in range(total_games):
        print(f"\nStarting game {i+1} of {total_games}...")
        result = play_game(algorithm, use_alpha_beta)
        if result == "user":
            user_score += 1
        elif result == "computer":
            computer_score += 1

        # Record game results: [game number, result, cumulative user score, cumulative computer score]
        results.append([i+1, result, user_score, computer_score])
        print("\nCurrent Score:")
        print("User (Baseline):", user_score, "Computer:", computer_score)

    print("\nFinal Score:")
    print("User (Baseline):", user_score, "Computer:", computer_score)
    print("Thanks for playing!")

    # Save results to CSV and generate graph.
    save_results(results, params)

if __name__ == '__main__':
    main()