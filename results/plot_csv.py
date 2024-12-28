import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read CSV file
file_path = "training_results_q_learning.csv"
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