import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_wins_q_sarsa(file_path='training_results_q_learning.csv'):
    df = pd.read_csv(file_path)

    # Create a FacetGrid to visualize the data
    g = sns.FacetGrid(
        df, col="Exploration Probability", row="Learning Rate", margin_titles=True, height=3.5
    )
    g.map(sns.lineplot, "Discount Factor", "Percentage Win Agent", marker="o")

    # Adjust titles and labels
    g.set_titles(col_template="Exploration Prob={col_name}", row_template="Learning Rate={row_name}")
    g.set_axis_labels("Discount Factor", "Percentage Win Agent")
    g.set(ylim=(50, 60))  # Set y-axis limits for better visualization
    g.tight_layout()  # Adjust layout to prevent overlap

    # Display the plots
    plt.show()

def plot_wins_discount_factor_dqn(file_path="training_results_dq_learning.csv"):
    df = pd.read_csv(file_path)

    discount_factors = df["Discount Factor"]
    agent_win_percentage = df["Percentage Win Agent"]

    plt.figure(figsize=(8, 5))
    plt.plot(discount_factors, agent_win_percentage, marker='o', linestyle='-', color='steelblue', label='Agent Win Percentage')

    plt.title('Agent Win Percentage vs Discount Factor', fontsize=14)
    plt.xlabel('Discount Factor', fontsize=12)
    plt.ylabel('Percentage Win Agent (%)', fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(discount_factors)
    plt.legend()
    plt.show()

plot_wins_discount_factor_dqn()