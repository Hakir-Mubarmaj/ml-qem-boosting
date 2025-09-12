# plot_sweep.py
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('results/smoke_sweep/sweep_results._updated.csv')  # or original if updated in-place
df['test_rmse'] = pd.to_numeric(df['test_rmse'], errors='coerce')
pivot = df.pivot_table(index='n_features', columns='model', values='test_rmse', aggfunc='mean')
pivot.plot(marker='o', linewidth=2)
plt.xlabel('n_features')
plt.ylabel('test RMSE')
plt.title('Feature sweep: RMSE vs n_features')
plt.grid(True)
plt.savefig('results/smoke_sweep/rmse_vs_features.png', dpi=150)
plt.show()
