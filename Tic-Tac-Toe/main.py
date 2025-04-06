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
    start_time = time.time()
    
    if algorithm == "baseline":
        move = baseline.baseline_move(game, player_letter)
    elif algorithm == "minimax":
        if use_alpha_beta:
            move = minimax.minimax(game, player_letter, -float('inf'), float('inf'))["position"]
        else:
            move = minimax.minimax_no_ab(game, player_letter)["position"]
    elif algorithm == "qlearning":
        move = qlearning.q_learning_move(game, player_letter)
    else:
        move = random.choice(game.available_moves())
    
    elapsed_time = time.time() - start_time
    return move, elapsed_time

def play_game_matchup(matchup, use_alpha_beta):
    game = TicTacToe()
    if matchup == "1":
        algo1, algo2 = "baseline", "minimax"
    elif matchup == "2":
        algo1, algo2 = "baseline", "qlearning"
    elif matchup == "3":
        algo1, algo2 = "minimax", "qlearning"
    elif matchup == "4":
        algo1, algo2 = "qlearning", "minimax"
    else:
        algo1, algo2 = "baseline", "minimax"

    player1_letter, player2_letter = 'X', 'O'
    turn = "player1" if random.random() < 0.5 else "player2"
    
    moves_count = 0
    algo1_total_time = 0
    algo2_total_time = 0
    algo1_moves = 0
    algo2_moves = 0

    while game.empty_squares():
        moves_count += 1
        
        if turn == "player1":
            move, move_time = get_move(game, player1_letter, algo1, use_alpha_beta)
            algo1_total_time += move_time
            algo1_moves += 1
            
            game.make_move(move, player1_letter)
            if game.current_winner == player1_letter:
                if algo1 == "qlearning": qlearning.update_terminal(10)
                if algo2 == "qlearning": qlearning.update_terminal(-10)
            
                algo1_avg_time = algo1_total_time / algo1_moves if algo1_moves > 0 else 0
                algo2_avg_time = algo2_total_time / algo2_moves if algo2_moves > 0 else 0
                
                return "player1", algo1, algo2, algo1_avg_time, algo2_avg_time, moves_count
            turn = "player2"
        else:
            move, move_time = get_move(game, player2_letter, algo2, use_alpha_beta)
            algo2_total_time += move_time
            algo2_moves += 1
            
            game.make_move(move, player2_letter)
            if game.current_winner == player2_letter:
                if algo2 == "qlearning": qlearning.update_terminal(10)
                if algo1 == "qlearning": qlearning.update_terminal(-10)
                
                algo1_avg_time = algo1_total_time / algo1_moves if algo1_moves > 0 else 0
                algo2_avg_time = algo2_total_time / algo2_moves if algo2_moves > 0 else 0
                
                return "player2", algo1, algo2, algo1_avg_time, algo2_avg_time, moves_count
            turn = "player1"

    if algo1 == "qlearning": qlearning.update_terminal(0)
    if algo2 == "qlearning": qlearning.update_terminal(0)

    algo1_avg_time = algo1_total_time / algo1_moves if algo1_moves > 0 else 0
    algo2_avg_time = algo2_total_time / algo2_moves if algo2_moves > 0 else 0
    
    return "tie", algo1, algo2, algo1_avg_time, algo2_avg_time, moves_count

def play_user_vs_ai(ai_type="minimax", use_alpha_beta=True):
    game = TicTacToe()
    player_letter = 'X'
    ai_letter = 'O'
    turn = "user"

    if ai_type == "qlearning":
        qlearning.load_model()

    print("\nYou are X. AI is O.")
    game.print_board()

    while game.empty_squares():
        if turn == "user":
            valid = False
            while not valid:
                try:
                    user_move = int(input("Enter your move (0-8): ").strip())
                    if user_move in game.available_moves():
                        valid = True
                        game.make_move(user_move, player_letter)
                        game.print_board()
                        if game.current_winner == player_letter:
                            print("\nYou win!")
                            return
                        turn = "ai"
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Enter a number between 0 and 8.")
        else:
            print("\nAI is thinking...")
            ai_move, _ = get_move(game, ai_letter, ai_type, use_alpha_beta)
            game.make_move(ai_move, ai_letter)
            print(f"AI plays: {ai_move}")
            game.print_board()
            if game.current_winner == ai_letter:
                print("\nAI wins!")
                return
            turn = "user"

    print("\nIt's a tie!")

def save_results(results, parameters, algo1_times, algo2_times, moves_per_game, folder_prefix="tictactoe_results"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{folder_prefix}_{now}"
    os.makedirs(folder_name, exist_ok=True)

    avg_algo1_time = sum(algo1_times) / len(algo1_times) if algo1_times else 0
    avg_algo2_time = sum(algo2_times) / len(algo2_times) if algo2_times else 0
    avg_moves = sum(moves_per_game) / len(moves_per_game) if moves_per_game else 0
    
    csv_file = os.path.join(folder_name, "results.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Game Number", 
            "Winner", 
            f"{parameters['player1_algo']} Score", 
            f"{parameters['player2_algo']} Score",
            f"{parameters['player1_algo']} Avg Time (s)",
            f"{parameters['player2_algo']} Avg Time (s)",
            "Moves Count"
        ])
        for i, row in enumerate(results):
            writer.writerow(row + [algo1_times[i], algo2_times[i], moves_per_game[i]])
    print(f"CSV results saved to {csv_file}")
    
    games = range(1, len(results) + 1)
    algo1_scores = [row[2] for row in results]
    algo2_scores = [row[3] for row in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(games, algo1_scores, label=f"{parameters['player1_algo']} Score")
    plt.plot(games, algo2_scores, label=f"{parameters['player2_algo']} Score")
    plt.xlabel("Game Number")
    plt.ylabel("Cumulative Score")
    plt.title("Cumulative Score Over TicTacToe Games")
    plt.legend()
    line_chart_file = os.path.join(folder_name, "cumulative_scores_line.png")
    plt.savefig(line_chart_file)
    plt.close()
    print(f"Line chart saved to {line_chart_file}")
    
    final_scores = [parameters['player1_algo'], parameters['player2_algo']]
    scores = [algo1_scores[-1] if algo1_scores else 0, algo2_scores[-1] if algo2_scores else 0]
    plt.figure(figsize=(8, 6))
    plt.bar(final_scores, scores, color=["blue", "orange"])
    plt.xlabel("Algorithm")
    plt.ylabel("Final Cumulative Score")
    plt.title("Final Cumulative Scores")
    bar_chart_file = os.path.join(folder_name, "final_scores_bar.png")
    plt.savefig(bar_chart_file)
    plt.close()
    print(f"Bar chart saved to {bar_chart_file}")
    
    plt.figure(figsize=(8, 6))
    plt.bar(final_scores, [avg_algo1_time, avg_algo2_time], color=["blue", "orange"])
    plt.xlabel("Algorithm")
    plt.ylabel("Average Move Time (seconds)")
    plt.title("Average Move Time by Algorithm")
    time_chart_file = os.path.join(folder_name, "avg_move_times_bar.png")
    plt.savefig(time_chart_file)
    plt.close()
    print(f"Move time chart saved to {time_chart_file}")
    
    params_file = os.path.join(folder_name, "parameters_and_stats.txt")
    with open(params_file, "w") as pf:
        pf.write("Parameters used:\n")
        for key, value in parameters.items():
            pf.write(f"{key}: {value}\n")
        
        pf.write("\nStatistics:\n")
        pf.write(f"Average moves per game: {avg_moves:.2f}\n")
        pf.write(f"Average {parameters['player1_algo']} move time: {avg_algo1_time:.6f} seconds\n")
        pf.write(f"Average {parameters['player2_algo']} move time: {avg_algo2_time:.6f} seconds\n")
        
    print(f"Parameters and statistics saved to {params_file}")

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    print("\n=== TicTacToe Matchup Menu ===")
    print("1. Baseline vs Minimax")
    print("2. Baseline vs Q-Learning")
    print("3. Minimax vs Q-Learning")
    print("4. Q-Learning vs Minimax")
    print("5. Play against AI")
    print("q. Quit")
    return input("Enter your choice (1-5 or q): ").strip()

def main():
    clear_terminal()
    while True:
        choice = main_menu()
        if choice.lower() == 'q':
            break

        use_alpha_beta = False
        if choice in ["1", "3", "4"]:
            use_alpha_beta = select_alpha_beta()

        total_games = int(input("How many iterations (games)? "))

        if choice == "1":
            player1_algo, player2_algo = "baseline", "minimax"
        elif choice == "2":
            player1_algo, player2_algo = "baseline", "qlearning"
        elif choice == "3":
            player1_algo, player2_algo = "minimax", "qlearning"
        elif choice == "4":
            player1_algo, player2_algo = "qlearning", "minimax"
        elif choice == "5":
            ai = input("Choose AI to play against (minimax or qlearning): ").strip().lower()
            ab = True
            if ai == "minimax":
                ab = select_alpha_beta()
            play_user_vs_ai(ai_type=ai, use_alpha_beta=ab)
            continue
        else:
            player1_algo, player2_algo = "baseline", "minimax"

        results = []
        algo1_times = []
        algo2_times = []
        moves_per_game = []
        score1, score2 = 0, 0
        
        params = {
            "matchup": choice,
            "use_alpha_beta": use_alpha_beta,
            "total_games": total_games,
            "ALPHA": qlearning.ALPHA if hasattr(qlearning, 'ALPHA') else "N/A",
            "GAMMA": qlearning.GAMMA if hasattr(qlearning, 'GAMMA') else "N/A",
            "EPSILON": qlearning.EPSILON if hasattr(qlearning, 'EPSILON') else "N/A",
            "player1_algo": player1_algo,
            "player2_algo": player2_algo
        }

        start_time = time.time()
        print("\nRunning games...")
        for i in range(total_games):
            winner, algo1, algo2, algo1_time, algo2_time, moves_count = play_game_matchup(choice, use_alpha_beta)
        
            algo1_times.append(algo1_time)
            algo2_times.append(algo2_time)
            moves_per_game.append(moves_count)
            
            if winner == "player1": 
                score1 += 1
            elif winner == "player2": 
                score2 += 1
                
            results.append([i+1, winner, score1, score2])
            
            print(f"Game {i+1}: Winner = {winner} | {player1_algo}: {score1} - {player2_algo}: {score2}")
            print(f"  {player1_algo} avg time: {algo1_time:.6f}s | {player2_algo} avg time: {algo2_time:.6f}s | Moves: {moves_count}")

        elapsed_time = time.time() - start_time
        print(f"\nFinal Score: {player1_algo} = {score1}, {player2_algo} = {score2}")
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        avg_algo1_time = sum(algo1_times) / len(algo1_times) if algo1_times else 0
        avg_algo2_time = sum(algo2_times) / len(algo2_times) if algo2_times else 0
        avg_moves = sum(moves_per_game) / len(moves_per_game) if moves_per_game else 0
        
        print(f"\nOverall Statistics:")
        print(f"Average moves per game: {avg_moves:.2f}")
        print(f"Average {player1_algo} move time: {avg_algo1_time:.6f} seconds")
        print(f"Average {player2_algo} move time: {avg_algo2_time:.6f} seconds")
        save_results(results, params, algo1_times, algo2_times, moves_per_game, folder_prefix="tictactoe_results")
        
        qlearning.save_model()
        print("\nSession complete.\n")

if __name__ == '__main__':
    main()