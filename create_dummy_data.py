import pandas as pd
import numpy as np

# Generate dummy Credit Card Fraud dataset
# Columns: Time, V1...V28, Amount, Class

n_samples = 1000
data = {
    'Time': np.arange(n_samples),
    'Amount': np.random.uniform(0, 1000, n_samples),
    'Class': np.concatenate([np.zeros(950), np.ones(50)])  # 5% fraud
}

# Generate V features
for i in range(1, 29):
    data[f'V{i}'] = np.random.normal(0, 1, n_samples)

df = pd.DataFrame(data)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('uploads/creditcard_sample.csv', index=False)
print("Created uploads/creditcard_sample.csv")
