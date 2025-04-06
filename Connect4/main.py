import random
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import csv

from game import Connect4
from algorithms import minimax, qlearning, baseline

def select_alpha_beta():
    response = input("Use alpha-beta pruning for minimax? (y/n): ").strip().lower()
    return response == 'y'

def get_move(game, player_letter, algorithm, use_alpha_beta, depth=4, time_limit=1800):
    if algorithm == "baseline":
        return baseline.baseline_move_connect4(game, player_letter)
    elif algorithm == "minimax":
        if use_alpha_beta:
            move_info = minimax.minimax_connect4(game, player_letter, depth, -float('inf'), float('inf'),
                                                 start_time=time.time(), time_limit=time_limit)
            return move_info["position"]
        else:
            move_info = minimax.minimax_no_ab_connect4(game, player_letter, depth,
                                                     start_time=time.time(), time_limit=time_limit)
            return move_info["position"]
    elif algorithm == "qlearning":
        return qlearning.q_learning_move_connect4(game, player_letter)
    else:
        return random.choice(game.available_moves())

def play_game_matchup(matchup, use_alpha_beta, depth=4):
    """Play a game between two algorithms with improved Q-learning integration."""
    # Create a new Connect4 game.
    game = Connect4()
    
    # Determine algorithms for the two sides.
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
    
    # In Connect4, we use letters "X" and "O".
    player1_letter = 'X'
    player2_letter = 'O'
    
    print("\nNew Connect4 game!")
    
    # Randomly decide who goes first.
    turn = "side1" if random.random() < 0.5 else "side2"
    
    moves_count = 0
    algo1_time = 0
    algo2_time = 0
    
    while game.empty_squares():
        moves_count += 1
        
        if turn == "side1":
            start_time = time.time()
            move = get_move(game, player1_letter, algo1, use_alpha_beta, depth)
            algo1_time += time.time() - start_time
            print(f"\n{algo1} chooses move: {move}")
            game.make_move(move, player1_letter)
            
            if game.current_winner == player1_letter:
                print(f"{algo1} wins!")
                return algo1, algo2, algo1, algo1_time, algo2_time, moves_count  # Winner is algo1.
                
            turn = "side2"
            
        else:  # side2's turn
            start_time = time.time()
            move = get_move(game, player2_letter, algo2, use_alpha_beta, depth)
            algo2_time += time.time() - start_time
            print(f"\n{algo2} chooses move: {move}")
            game.make_move(move, player2_letter)
            
            if game.current_winner == player2_letter:
                print(f"{algo2} wins!")
                return algo1, algo2, algo2, algo1_time, algo2_time, moves_count  # Winner is algo2.
                
            turn = "side1"
    
    # It's a tie (board full)
    print("It's a tie!")
    return algo1, algo2, "tie", algo1_time, algo2_time, moves_count

def play_vs_human(ai_type, use_alpha_beta, depth=4):
    game = Connect4()
    player_letter = 'X'
    ai_letter = 'O'

    print("\nYou are 'X'. The AI is 'O'. Let's play Connect4!")
    game.print_board()
    turn = "human" if random.random() < 0.5 else "ai"

    while game.empty_squares():
        if turn == "human":
            valid = False
            while not valid:
                try:
                    col = int(input("Your move (0-6): "))
                    if col in game.available_moves():
                        valid = True
                        game.make_move(col, player_letter)
                        print(f"You placed in column {col}")
                    else:
                        print("Column full or invalid. Try again.")
                except ValueError:
                    print("Invalid input. Enter a number between 0-6.")
        else:
            print("AI is thinking...")
            move = get_move(game, ai_letter, ai_type, use_alpha_beta, depth)  # No need to unpack
            game.make_move(move, ai_letter)
            print(f"AI placed in column {move}")

        game.print_board()

        if game.current_winner:
            if turn == "human":
                print("🎉 You win!")
                return
            else:
                print("💻 AI wins!")
                return

        turn = "ai" if turn == "human" else "human"

    print("It's a draw!")

    print("It's a draw!")
def save_results(results, parameters, algo1_times=None, algo2_times=None, moves_per_game=None, start_time=None, folder_prefix="connect4_results"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{folder_prefix}_{now}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Save CSV file (Append data)
    csv_file = os.path.join(folder_name, "results.csv")
    file_exists = os.path.exists(csv_file)
    
    # Calculate overall statistics
    total_games = len(results)
    total_moves = sum(moves_per_game)
    average_moves = total_moves / total_games if total_games > 0 else 0
    avg_algo1_time = sum(algo1_times) / total_games if total_games > 0 else 0
    avg_algo2_time = sum(algo2_times) / total_games if total_games > 0 else 0
    score_algo1 = sum([1 for result in results if result[1] == parameters['player1_algo']])
    score_algo2 = total_games - score_algo1  # Assuming the rest of the games are won by algo2 or tie

    # Write the header if the file is being created
    with open(csv_file, mode='a' if file_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Match No", "Use Alpha-Beta", "Total Games", "MINMAX-ALPHA", "MINMAX-BETA", 
                "QL-ALPHA", "QL-GAMMA", "QL-EPSILON", "Player 1 Algorithm", "Player 2 Algorithm", 
                "Player 1 Final Score", "Player 2 Final Score", "Tie Score", 
                "Total Execution Time (s)", "Average Moves per Game", 
                "Average Player 1 Move Time (s)", "Average Player 2 Move Time (s)"
            ])
        
        writer.writerow([
            total_games, 
            "TRUE" if parameters['use_alpha_beta'] else "-",
            total_games, 
            parameters.get('MINMAX-ALPHA', '-'),
            parameters.get('MINMAX-BETA', '-'),
            parameters.get('QL-ALPHA', '-'),
            parameters.get('QL-GAMMA', '-'),
            parameters.get('QL-EPSILON', '-'),
            parameters['player1_algo'], 
            parameters['player2_algo'], 
            score_algo1, 
            score_algo2, 
            total_games - score_algo1 - score_algo2, 
            time.time() - start_time,
            average_moves, 
            avg_algo1_time, 
            avg_algo2_time
        ])
    
    print(f"CSV results saved to {csv_file}")
    
    # Save parameters and overall statistics
    stats_file = os.path.join(folder_name, "parameters_and_stats.txt")
    with open(stats_file, "w") as pf:
        pf.write(f"Parameters used for this run:\n")
        for key, value in parameters.items():
            pf.write(f"{key}: {value}\n")
        
        pf.write(f"\nOverall Statistics:\n")
        pf.write(f"Total games played: {total_games}\n")
        pf.write(f"Average moves per game: {average_moves:.2f}\n")
        pf.write(f"Average move time for {parameters['player1_algo']}: {avg_algo1_time:.6f} seconds\n")
        pf.write(f"Average move time for {parameters['player2_algo']}: {avg_algo2_time:.6f} seconds\n")
    print(f"Parameters and statistics saved to {stats_file}")
    
    # Generate line chart for scores
    games = list(range(1, total_games + 1))
    algo1_scores = [score_algo1] * total_games
    algo2_scores = [score_algo2] * total_games
    
    plt.figure(figsize=(10, 6))
    plt.plot(games, algo1_scores, label=f"{parameters['player1_algo']} Score")
    plt.plot(games, algo2_scores, label=f"{parameters['player2_algo']} Score")
    plt.xlabel("Game Number")
    plt.ylabel("Cumulative Score")
    plt.title("Line Chart: Cumulative Score Over Connect4 Games")
    plt.legend()
    line_chart_file = os.path.join(folder_name, "cumulative_scores_line.png")
    plt.savefig(line_chart_file)
    plt.close()
    print(f"Line chart saved to {line_chart_file}")
    
    # Generate average move time chart.
    plt.figure(figsize=(8, 6))
    plt.bar([parameters['player1_algo'], parameters['player2_algo']], [avg_algo1_time, avg_algo2_time], color=["blue", "orange"])
    plt.xlabel("Algorithm")
    plt.ylabel("Average Move Time (seconds)")
    plt.title("Average Move Time for Algorithms")
    move_time_chart_file = os.path.join(folder_name, "avg_move_times_bar.png")
    plt.savefig(move_time_chart_file)
    plt.close()
    print(f"Move time chart saved to {move_time_chart_file}")
def main_menu():
    print("\n=== Connect4 Menu ===")
    print("1. Baseline vs Minimax")
    print("2. Baseline vs Q-Learning")
    print("3. Minimax vs Q-Learning")
    print("4. Q-Learning vs Minimax")
    print("5. Play against AI (Q-Learning or Minimax)")
    print("q. Quit")
    return input("Enter your choice (1-5 or q): ").strip()

def main():
    while True:
        choice = main_menu()
        if choice.lower() == 'q':
            break

        if choice == "5":
            print("\nChoose your AI opponent:")
            print("1. Q-Learning")
            print("2. Minimax")
            ai_choice = input("Enter 1 or 2: ").strip()
            ai_algo = "qlearning" if ai_choice == "1" else "minimax"
            use_alpha_beta = False
            depth = 4
            if ai_algo == "minimax":
                use_alpha_beta = select_alpha_beta()
                depth = int(input("Enter depth for Minimax: "))
            play_vs_human(ai_algo, use_alpha_beta, depth)
            continue

        # Matchup games for training/evaluation
        use_alpha_beta = False
        if choice in ["1", "3", "4"]:
            use_alpha_beta = select_alpha_beta()
        depth = 4
        if choice in ["1", "3", "4"]:
            depth = int(input("Enter depth limit for Minimax: "))
        total_games = int(input("How many iterations (games) to run? "))

        if choice == "1":
            player1_algo, player2_algo = "baseline", "minimax"
        elif choice == "2":
            player1_algo, player2_algo = "baseline", "qlearning"
        elif choice == "3":
            player1_algo, player2_algo = "minimax", "qlearning"
        elif choice == "4":
            player1_algo, player2_algo = "qlearning", "minimax"
        else:
            player1_algo, player2_algo = "baseline", "minimax"

        results = []
        algo1_times = []
        algo2_times = []
        moves_per_game = []
        score_algo1 = 0
        score_algo2 = 0

        params = {
            "matchup": choice,
            "use_alpha_beta": use_alpha_beta,
            "depth": depth,
            "total_games": total_games,
            "player1_algo": player1_algo,
            "player2_algo": player2_algo
        }

        start_time = time.time()
        for i in range(total_games):
            print(f"\n🔁 Game {i+1}/{total_games}")
            algo1_used, algo2_used, winner, algo1_time, algo2_time, moves_count = play_game_matchup(choice, use_alpha_beta, depth)
            algo1_times.append(algo1_time)
            algo2_times.append(algo2_time)
            moves_per_game.append(moves_count)

            if winner == algo1_used:
                score_algo1 += 1
            elif winner == algo2_used:
                score_algo2 += 1

            results.append([i+1, winner, score_algo1, score_algo2])
            print(f"Score -> {player1_algo}: {score_algo1}, {player2_algo}: {score_algo2}")
        
        elapsed_time = time.time() - start_time
        print(f"\n📊 Session Complete!")
        print(f"⏱️ Total time: {elapsed_time:.2f}s")
        print(f"🏁 Final Score: {player1_algo}: {score_algo1}, {player2_algo}: {score_algo2}")

        # Save results and graphs, as before
        save_results(results, params, algo1_times, algo2_times, moves_per_game)

if __name__ == '__main__':
    main()