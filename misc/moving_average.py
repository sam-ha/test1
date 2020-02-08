mylist = [1, 2, 3, 4, 5, 6, 7]
N = 3
cumsum, moving_aves = [0], []

for i, x in enumerate(mylist, 1):
    cumsum.append(cumsum[i-1] + x)
    print(cumsum)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)

import numpy as np
def movingaverage(interval, window_size):
  window = np.ones(int(window_size)) / float(window_size)
  return np.convolve(interval, window, 'same')

mov=movingaverage(mylist, 3)
print(moving_aves)
print(mov)