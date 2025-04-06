import matplotlib.pyplot as plt

# Data for the total games and average moves per game and algorithm moves
total_games = [10, 100, 1000]  # Total number of games

# Data extracted from the table for algorithms (Average Moves to Complete Game)
avg_moves_to_complete_game = [25, 24.99, 25.739]  # Total moves to complete a game for 10, 100, 1000 games

# Baseline average moves per game for player 1 and player 2
baseline_avg_moves = [23.5, 25.42, 25.449]  # Baseline average moves for 10, 100, 1000 games

# Minimax average moves per game for player 1 and player 2
minimax_avg_moves = [26.9, 28.25, 34.714]  # Minimax average moves for 10, 100, 1000 games

# QLearning average moves per game for player 1 and player 2
qlearning_avg_moves = [29.5, 27.94, 34.667]  # QLearning average moves for 10, 100, 1000 games

# Create figure and axis
plt.figure(figsize=(10, 6))

# Plotting the line graph for each algorithm
plt.plot(total_games, avg_moves_to_complete_game, label='Total Moves to Complete Game', marker='o', color='black')
plt.plot(total_games, baseline_avg_moves, label='Baseline', marker='o', color='blue')
plt.plot(total_games, minimax_avg_moves, label='Minimax', marker='o', color='orange')
plt.plot(total_games, qlearning_avg_moves, label='Q-Learning', marker='o', color='green')

# Adding labels and title
plt.title('Total Moves to Complete Game vs Algorithm Moves to Win')
plt.xlabel('Total Games')
plt.ylabel('Average Moves')
plt.legend(title="Algorithm")
plt.grid(True)

# Display the plot
plt.show()