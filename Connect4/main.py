import random
import os
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt

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
    
    while game.empty_squares():
        moves_count += 1
        
        if turn == "side1":
            move = get_move(game, player1_letter, algo1, use_alpha_beta, depth)
            print(f"\n{algo1} chooses move: {move}")
            game.make_move(move, player1_letter)
            
            if game.current_winner == player1_letter:
                print(f"{algo1} wins!")
                
                # Reward structure for Q-learning
                if algo1 == "qlearning":
                    # Higher reward for winning quickly
                    win_reward = 50 - (0.3 * moves_count)  
                    qlearning.update_terminal_connect4(max(10, win_reward))
                if algo2 == "qlearning":
                    # Loss penalty depends on game length
                    loss_penalty = -30 if moves_count < 10 else -10
                    qlearning.update_terminal_connect4(loss_penalty)
                    
                return algo1, algo2, algo1  # Winner is algo1.
                
            turn = "side2"
            
        else:  # side2's turn
            move = get_move(game, player2_letter, algo2, use_alpha_beta, depth)
            print(f"\n{algo2} chooses move: {move}")
            game.make_move(move, player2_letter)
            
            if game.current_winner == player2_letter:
                print(f"{algo2} wins!")
                
                # Reward structure for Q-learning
                if algo2 == "qlearning":
                    # Higher reward for winning quickly
                    win_reward = 50 - (0.3 * moves_count)
                    qlearning.update_terminal_connect4(max(10, win_reward))
                if algo1 == "qlearning":
                    # Loss penalty depends on game length
                    loss_penalty = -30 if moves_count < 10 else -10
                    qlearning.update_terminal_connect4(loss_penalty)
                    
                return algo1, algo2, algo2  # Winner is algo2.
                
            turn = "side1"
    
    # It's a tie (board full)
    print("It's a tie!")
    
    # Small penalty for ties
    if algo1 == "qlearning":
        qlearning.update_terminal_connect4(-5)
    if algo2 == "qlearning":
        qlearning.update_terminal_connect4(-5)
        
    return algo1, algo2, "tie"

def save_results(results, parameters, folder_prefix="connect4_results"):
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
    plt.title("Line Chart: Cumulative Score Over Connect4 Games")
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

def play_vs_human(ai_type, use_alpha_beta, depth=4):
    game = Connect4()
    player_letter = 'X'
    ai_letter = 'O'

    qlearning.load_model() if ai_type == 'qlearning' else None

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
            move, _ = get_move(game, ai_letter, ai_type, use_alpha_beta, depth)
            game.make_move(move, ai_letter)
            print(f"AI placed in column {move}")

        game.print_board()

        if game.current_winner:
            if turn == "human":
                print("🎉 You win!")
                if ai_type == "qlearning":
                    qlearning.update_terminal_connect4(-10)
                return
            else:
                print("💻 AI wins!")
                if ai_type == "qlearning":
                    qlearning.update_terminal_connect4(10)
                return

        turn = "ai" if turn == "human" else "human"

    print("It's a draw!")
    if ai_type == "qlearning":
        qlearning.update_terminal_connect4(-5)

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
            "ALPHA": qlearning.ALPHA,
            "GAMMA": qlearning.GAMMA,
            "EPSILON": qlearning.EPSILON,
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

        save_results(results, params, algo1_times, algo2_times, moves_per_game)

        if player1_algo == "qlearning" or player2_algo == "qlearning":
            qlearning.save_model()

if __name__ == '__main__':
    main()