import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

NEW_TIC_NAME = 'neg_sine'

out_dir = "/home/matthias/Projects/FinRL/datasets/raw/1d/"
df = pd.read_pickle("/home/matthias/Projects/FinRL/datasets/raw/1d/AAPL_2014-01-06_2022-12-13")

sine = [np.sin(-0.1 * x)*5+10 for x in range(len(df))]
line = [1 + 0.05 * x for x in range(len(df))]
linesine = [l+s for l, s in zip(line, sine)]
volume = [np.max([s, 0.1]) * 10 for s in sine]

print(df.columns)
for c in ['open', 'high', 'low', 'close']:
    df[c] = sine
    # df[c] = linesine

df['volume'] = volume

df.close.plot()
# df.volume.plot()
# plt.show()
# exit()
start = (df.iloc[0].date)
end = (df.iloc[-1].date)
df.tic = [NEW_TIC_NAME] * len(df)
pd.to_pickle(df, os.path.join(out_dir, '{}_{}_{}'.format(NEW_TIC_NAME, start,end)))

