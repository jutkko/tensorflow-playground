import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(
    np.random.randint(2, 8, 5000),
    columns = ['one'])
df['two'] = df['one'] + np.random.randint(2, 8, 5000)
ax = df.plot.hist(bins=12, alpha=0.5)
plt.show()