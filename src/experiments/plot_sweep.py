# plot_sweep.py
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/smoke_sweep/sweep_results.csv")

# convert n_features to numeric (if not)
df['n_features'] = pd.to_numeric(df['n_features'], errors='coerce')

plt.figure(figsize=(10, 6))  # ছবিকে wide করা হলো

for model in df['model'].unique():
    sub = df[df['model'] == model].sort_values('n_features')
    plt.plot(sub['n_features'], sub['val_score'], marker='o', label=f'{model} val')
    plt.plot(sub['n_features'], sub['test_rmse'], marker='x', linestyle='--', label=f'{model} test')

plt.xlabel('n_features')
plt.ylabel('error (lower = better)')
plt.title('Feature sweep: val_score vs test_rmse')
plt.grid(True)

# legend কে ডান পাশে সরানো হলো
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

out = "results/smoke_sweep/feature_sweep_plot.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print("Saved plot to", out)
plt.show()
