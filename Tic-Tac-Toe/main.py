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

def get_move(game, player_letter, algorithm, use_alpha_beta, time_limit=30):
    if algorithm == "baseline":
        return baseline.baseline_move(game, player_letter)
    elif algorithm == "minimax":
        # For TicTacToe, full game search is usually feasible.
        if use_alpha_beta:
            move_info = minimax.minimax(game, player_letter, alpha=-float('inf'), beta=float('inf'))
            return move_info["position"]
        else:
            move_info = minimax.minimax_no_ab(game, player_letter)
            return move_info["position"]
    elif algorithm == "qlearning":
        return qlearning.q_learning_move(game, player_letter)
    else:
        return random.choice(game.available_moves())

def play_game_matchup(matchup, use_alpha_beta):
    # Create a new TicTacToe game.
    game = TicTacToe()
    # Determine algorithms for player1 and player2.
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
    
    print("\nNew TicTacToe game!")
    game.print_board()
    
    # Randomly decide who goes first.
    turn = "player1" if random.random() < 0.5 else "player2"
    
    while game.empty_squares():
        if turn == "player1":
            move = get_move(game, player1_letter, algo1, use_alpha_beta)
            print(f"\nPlayer1 ({algo1}) chooses move: {move}")
            game.make_move(move, player1_letter)
            game.print_board()
            if game.current_winner == player1_letter:
                print("Player1 wins!")
                if algo1 == "qlearning":
                    qlearning.update_terminal(10)
                if algo2 == "qlearning":
                    qlearning.update_terminal(-10)
                return "player1", algo1, algo2
            turn = "player2"
        else:
            move = get_move(game, player2_letter, algo2, use_alpha_beta)
            print(f"\nPlayer2 ({algo2}) chooses move: {move}")
            game.make_move(move, player2_letter)
            game.print_board()
            if game.current_winner == player2_letter:
                print("Player2 wins!")
                if algo2 == "qlearning":
                    qlearning.update_terminal(10)
                if algo1 == "qlearning":
                    qlearning.update_terminal(-10)
                return "player2", algo1, algo2
            turn = "player1"
    print("It's a tie!")
    if algo1 == "qlearning":
        qlearning.update_terminal(0)
    if algo2 == "qlearning":
        qlearning.update_terminal(0)
    return "tie", algo1, algo2

def save_results(results, parameters, folder_prefix="tictactoe_results"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{folder_prefix}_{now}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Save CSV file.
    csv_file = os.path.join(folder_name, "results.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Game Number", "Result", f"{parameters['player1_algo']} Score", f"{parameters['player2_algo']} Score"])
        for row in results:
            writer.writerow(row)
    print(f"CSV results saved to {csv_file}")
    
    # Generate line chart.
    games = [row[0] for row in results]
    algo1_scores = [row[2] for row in results]
    algo2_scores = [row[3] for row in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(games, algo1_scores, label=f"{parameters['player1_algo']} Score")
    plt.plot(games, algo2_scores, label=f"{parameters['player2_algo']} Score")
    plt.xlabel("Game Number")
    plt.ylabel("Cumulative Score")
    plt.title("Line Chart: Cumulative Score Over TicTacToe Games")
    plt.legend()
    line_chart_file = os.path.join(folder_name, "cumulative_scores_line.png")
    plt.savefig(line_chart_file)
    plt.close()
    print(f"Line chart saved to {line_chart_file}")
    
    # Generate bar chart.
    final_scores = [parameters['player1_algo'], parameters['player2_algo']]
    scores = [algo1_scores[-1] if algo1_scores else 0, algo2_scores[-1] if algo2_scores else 0]
    plt.figure(figsize=(8, 6))
    plt.bar(final_scores, scores, color=["blue", "orange"])
    plt.xlabel("Algorithm")
    plt.ylabel("Final Cumulative Score")
    plt.title("Bar Chart: Final Cumulative Scores")
    bar_chart_file = os.path.join(folder_name, "final_scores_bar.png")
    plt.savefig(bar_chart_file)
    plt.close()
    print(f"Bar chart saved to {bar_chart_file}")
    
    # Save parameters.
    params_file = os.path.join(folder_name, "parameters.txt")
    with open(params_file, "w") as pf:
        pf.write("Parameters used:\n")
        for key, value in parameters.items():
            pf.write(f"{key}: {value}\n")
    print(f"Parameters saved to {params_file}")

def main_menu():
    print("\n=== TicTacToe Matchup Menu ===")
    print("Select matchup type:")
    print("1. Baseline vs Minimax")
    print("2. Baseline vs Q-Learning")
    print("3. Minimax vs Q-Learning")
    print("4. Q-Learning vs Minimax")
    print("q. Quit")
    return input("Enter your choice (1-4 or q): ").strip()

def main():
    # Optionally, load a previously saved Q-learning model:
    # qlearning.load_model()
    
    while True:
        choice = main_menu()
        if choice.lower() == 'q':
            print("Exiting program.")
            break
        
        use_alpha_beta = False
        if choice in ["1", "3", "4"]:
            use_alpha_beta = select_alpha_beta()
        
        total_games = int(input("How many iterations (games) do you want to run? "))
        
        # Determine algorithm names based on matchup.
        if choice == "1":
            player1_algo = "baseline"
            player2_algo = "minimax"
        elif choice == "2":
            player1_algo = "baseline"
            player2_algo = "qlearning"
        elif choice == "3":
            player1_algo = "minimax"
            player2_algo = "qlearning"
        elif choice == "4":
            player1_algo = "qlearning"
            player2_algo = "minimax"
        else:
            player1_algo = "baseline"
            player2_algo = "minimax"
        
        results = []
        score_algo1 = 0
        score_algo2 = 0
        
        params = {
            "matchup": choice,
            "use_alpha_beta": use_alpha_beta,
            "total_games": total_games,
            "player1_algo": player1_algo,
            "player2_algo": player2_algo,
            "ALPHA": qlearning.ALPHA,
            "GAMMA": qlearning.GAMMA,
            "EPSILON": qlearning.EPSILON
        }
        
        start_time = time.time()
        for i in range(total_games):
            print(f"\nStarting TicTacToe game {i+1} of {total_games}...")
            result, algo1_used, algo2_used = play_game_matchup(choice, use_alpha_beta)
            if result == "player1":
                score_algo1 += 1
            elif result == "player2":
                score_algo2 += 1
            
            results.append([i+1, result, score_algo1, score_algo2])
            print(f"Current Score: {player1_algo}: {score_algo1}, {player2_algo}: {score_algo2}")
            # No per-game prompt; the games run continuously.
        
        elapsed_time = time.time() - start_time
        print(f"\nFinal Score for current session: {player1_algo}: {score_algo1}, {player2_algo}: {score_algo2}")
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        
        save_results(results, params, folder_prefix="tictactoe_results")
        print("Session complete.\n")
        
        # Optionally, offer to load a saved model for next session:
        # load_choice = input("Press 'l' to load saved Q-learning model for next session, or press Enter to continue: ").strip().lower()
        # if load_choice == 'l':
        #     qlearning.load_model()

if __name__ == '__main__':
    main()