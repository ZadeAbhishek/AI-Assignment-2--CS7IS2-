import matplotlib.pyplot as plt

# Data for the plot
total_matches = [10, 100, 1000]
baseline_vs_minmax = [25, 24.99, 25.739]
baseline_vs_minmax_without_alpha_beta = [23.4,25.42,25.449]
baseline_vs_qlearning = [22.3,26.28,26.748]
minmax_vs_qlearning = [26.9, 28.35, 34.714]
minmax_without_alpha_beta_vs_qlearning = [29.5, 27.94, 34.667]

# Plotting the graphs for Average Moves per Game vs Total Matches
plt.figure(figsize=(10, 6))

# Line plots for each dataset
plt.plot(total_matches, baseline_vs_minmax, label="Baseline vs Minimax", marker='o', color='blue')
plt.plot(total_matches, baseline_vs_minmax_without_alpha_beta, label="Baseline vs Minimax (without alpha-beta)", marker='o', color='orange')
plt.plot(total_matches, baseline_vs_qlearning, label="Baseline vs Q-Learning", marker='o', color='green')
plt.plot(total_matches, minmax_vs_qlearning, label="Minimax vs Q-Learning", marker='o', color='red')
plt.plot(total_matches, minmax_without_alpha_beta_vs_qlearning, label="Minimax (without alpha-beta) vs Q-Learning", marker='o', color='purple')

# Adding labels and title
plt.title('Average Moves per Game vs Total Matches')
plt.xlabel('Total Matches')
plt.ylabel('Average Moves per Game')
plt.legend(title="Matchup")
plt.grid(True)

# Display the plot
plt.show()