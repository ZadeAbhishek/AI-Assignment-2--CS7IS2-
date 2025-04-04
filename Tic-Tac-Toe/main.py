import random
import os
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt

from game import TicTacToe
from algorithms import minimax, qlearning, baseline

def select_alpha_beta():
    response = input("Use alpha-beta pruning for minimax? (y/n): ").strip().lower()
    return response == 'y'

def get_move(game, player_letter, algorithm, use_alpha_beta):
    if algorithm == "baseline":
        return baseline.baseline_move(game, player_letter)
    elif algorithm == "minimax":
        if use_alpha_beta:
            move_info = minimax.minimax(game, player_letter)
            return move_info["position"]
        else:
            move_info = minimax.minimax_no_ab(game, player_letter)
            return move_info["position"]
    elif algorithm == "qlearning":
        return qlearning.q_learning_move(game, player_letter)
    else:
        return random.choice(game.available_moves())

def play_game_matchup(matchup, use_alpha_beta):
    # Create a new game.
    game = TicTacToe()
    # Define which algorithm each player uses.
    if matchup == "1":   # Baseline vs Minimax
        algo1 = "baseline"
        algo2 = "minimax"
    elif matchup == "2":  # Baseline vs Q-Learning
        algo1 = "baseline"
        algo2 = "qlearning"
    elif matchup == "3":  # Minimax vs Q-Learning
        algo1 = "minimax"
        algo2 = "qlearning"
    elif matchup == "4":  # Q-Learning vs Minimax
        algo1 = "qlearning"
        algo2 = "minimax"
    else:
        algo1 = "baseline"
        algo2 = "minimax"
    
    player1_letter = 'X'
    player2_letter = 'O'
    
    print("\nNew game!")
    game.print_board()
    
    turn = "player1" if random.random() < 0.5 else "player2"
    
    while game.empty_squares():
        if turn == "player1":
            move = get_move(game, player1_letter, algo1, use_alpha_beta)
            print(f"\nPlayer 1 ({algo1}) chooses move: {move}")
            game.make_move(move, player1_letter)
            game.print_board()
            if game.current_winner == player1_letter:
                print("Player 1 wins!")
                if algo1 == "qlearning":
                    qlearning.update_terminal(1)
                if algo2 == "qlearning":
                    qlearning.update_terminal(-1)
                return "player1"
            turn = "player2"
        else:
            move = get_move(game, player2_letter, algo2, use_alpha_beta)
            print(f"\nPlayer 2 ({algo2}) chooses move: {move}")
            game.make_move(move, player2_letter)
            game.print_board()
            if game.current_winner == player2_letter:
                print("Player 2 wins!")
                if algo2 == "qlearning":
                    qlearning.update_terminal(1)
                if algo1 == "qlearning":
                    qlearning.update_terminal(-1)
                return "player2"
            turn = "player1"
    print("It's a tie!")
    if algo1 == "qlearning":
        qlearning.update_terminal(0)
    if algo2 == "qlearning":
        qlearning.update_terminal(0)
    return "tie"

def save_results(results, parameters):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"results_{now}"
    os.makedirs(folder_name, exist_ok=True)

    csv_file = os.path.join(folder_name, "results.csv")
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Game Number", "Result", "Player1 Score", "Player2 Score"])
        for row in results:
            writer.writerow(row)
    print(f"CSV results saved to {csv_file}")

    # Generate a line chart.
    games = [row[0] for row in results]
    p1_scores = [row[2] for row in results]
    p2_scores = [row[3] for row in results]
    plt.figure(figsize=(10, 6))
    plt.plot(games, p1_scores, label="Player1 Score")
    plt.plot(games, p2_scores, label="Player2 Score")
    plt.xlabel("Game Number")
    plt.ylabel("Cumulative Score")
    plt.title("Line Chart: Cumulative Score Over Games")
    plt.legend()
    line_chart_file = os.path.join(folder_name, "cumulative_scores_line.png")
    plt.savefig(line_chart_file)
    plt.close()
    print(f"Line chart saved to {line_chart_file}")

    # Generate a bar chart for final scores.
    final_scores = ["Player1", "Player2"]
    scores = [p1_scores[-1] if p1_scores else 0, p2_scores[-1] if p2_scores else 0]
    plt.figure(figsize=(8, 6))
    plt.bar(final_scores, scores, color=["blue", "orange"])
    plt.xlabel("Players")
    plt.ylabel("Final Cumulative Score")
    plt.title("Bar Chart: Final Cumulative Scores")
    bar_chart_file = os.path.join(folder_name, "final_scores_bar.png")
    plt.savefig(bar_chart_file)
    plt.close()
    print(f"Bar chart saved to {bar_chart_file}")

    params_file = os.path.join(folder_name, "parameters.txt")
    with open(params_file, "w") as pf:
        pf.write("Parameters used:\n")
        for key, value in parameters.items():
            pf.write(f"{key}: {value}\n")
    print(f"Parameters saved to {params_file}")

def main():
    start_time = time.time()
    
    print("Select matchup type:")
    print("1. Baseline vs Minimax")
    print("2. Baseline vs Q-Learning")
    print("3. Minimax vs Q-Learning")
    print("4. Q-Learning vs Minimax")
    matchup = input("Enter your choice (1-4): ").strip()
    
    use_alpha_beta = False
    # If minimax is involved, ask for alpha-beta pruning.
    if matchup in ["1", "3", "4"]:
        use_alpha_beta = select_alpha_beta()
    
    total_games = int(input("How many iterations (games) do you want to run? "))
    
    results = []
    p1_score = 0
    p2_score = 0
    
    params = {
        "matchup": matchup,
        "use_alpha_beta": use_alpha_beta,
        "total_games": total_games,
        "ALPHA": qlearning.ALPHA,
        "GAMMA": qlearning.GAMMA,
        "EPSILON": qlearning.EPSILON
    }
    
    for i in range(total_games):
        print(f"\nStarting game {i+1} of {total_games}...")
        result = play_game_matchup(matchup, use_alpha_beta)
        if result == "player1":
            p1_score += 1
        elif result == "player2":
            p2_score += 1
        
        results.append([i+1, result, p1_score, p2_score])
        print(f"Current Score: Player1: {p1_score}, Player2: {p2_score}")
    
    print("\nFinal Score:")
    print(f"Player1: {p1_score}, Player2: {p2_score}")
    
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    
    save_results(results, params)

if __name__ == '__main__':
    main()