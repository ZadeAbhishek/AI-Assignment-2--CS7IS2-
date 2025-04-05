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
        if use_alpha_beta:
            return minimax.minimax(game, player_letter, -float('inf'), float('inf'))["position"]
        else:
            return minimax.minimax_no_ab(game, player_letter)["position"]
    elif algorithm == "qlearning":
        return qlearning.q_learning_move(game, player_letter)
    else:
        return random.choice(game.available_moves())

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

    while game.empty_squares():
        if turn == "player1":
            move = get_move(game, player1_letter, algo1, use_alpha_beta)
            game.make_move(move, player1_letter)
            if game.current_winner == player1_letter:
                if algo1 == "qlearning": qlearning.update_terminal(10)
                if algo2 == "qlearning": qlearning.update_terminal(-10)
                return "player1", algo1, algo2
            turn = "player2"
        else:
            move = get_move(game, player2_letter, algo2, use_alpha_beta)
            game.make_move(move, player2_letter)
            if game.current_winner == player2_letter:
                if algo2 == "qlearning": qlearning.update_terminal(10)
                if algo1 == "qlearning": qlearning.update_terminal(-10)
                return "player2", algo1, algo2
            turn = "player1"

    if algo1 == "qlearning": qlearning.update_terminal(0)
    if algo2 == "qlearning": qlearning.update_terminal(0)
    return "tie", algo1, algo2

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    print("\n=== TicTacToe Matchup Menu ===")
    print("1. Baseline vs Minimax")
    print("2. Baseline vs Q-Learning")
    print("3. Minimax vs Q-Learning")
    print("4. Q-Learning vs Minimax")
    print("q. Quit")
    return input("Enter your choice (1-4 or q): ").strip()

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
        else:
            player1_algo, player2_algo = "baseline", "minimax"

        results = []
        score1, score2 = 0, 0

        print("\nRunning games...")
        for i in range(total_games):
            winner, algo1, algo2 = play_game_matchup(choice, use_alpha_beta)
            if winner == "player1": score1 += 1
            elif winner == "player2": score2 += 1
            print(f"Game {i+1}: Winner = {winner} | {player1_algo}: {score1} - {player2_algo}: {score2}")

        print(f"\nFinal Score: {player1_algo} = {score1}, {player2_algo} = {score2}")
        qlearning.save_model()

if __name__ == '__main__':
    main()