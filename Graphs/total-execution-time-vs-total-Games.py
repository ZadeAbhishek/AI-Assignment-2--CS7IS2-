import matplotlib.pyplot as plt

# Data for the plot
total_matches = [10, 100, 1000]  # Total number of games

# Data for the average execution time of different matchups
baseline_vs_minmax = [0.23, 2.34, 24.6]  # Baseline vs Minimax (execution time)
baseline_vs_minmax_without_alpha_beta = [7.26, 61.72, 592.96]  # Baseline vs Minimax (without alpha-beta)
baseline_vs_qlearning = [0.01, 0.03, 0.12]  # Baseline vs Q-learning (execution time)
minmax_vs_qlearning = [0.26, 2.66, 21.96]  # Minimax vs Q-learning (execution time)
minmax_without_alpha_beta_vs_qlearning = [4.59, 54.86, 546.48]  # Minimax (without alpha-beta) vs Q-learning (execution time)

# Plotting the graphs
plt.figure(figsize=(10, 6))

# Line plots for each dataset
plt.plot(total_matches, baseline_vs_minmax, label="Baseline vs Minimax", marker='o', color='blue')
plt.plot(total_matches, baseline_vs_minmax_without_alpha_beta, label="Baseline vs Minimax (without alpha-beta)", marker='o', color='orange')
plt.plot(total_matches, baseline_vs_qlearning, label="Baseline vs Q-Learning", marker='o', color='green')
plt.plot(total_matches, minmax_vs_qlearning, label="Minimax vs Q-Learning", marker='o', color='red')
plt.plot(total_matches, minmax_without_alpha_beta_vs_qlearning, label="Minimax (without alpha-beta) vs Q-Learning", marker='o', color='purple')

# Adding labels and title
plt.title('Average Execution Time vs Total Matches')
plt.xlabel('Total Matches')
plt.ylabel('Average Execution Time (s)')
plt.legend(title="Matchup")
plt.grid(True)

# Display the plot
plt.show()