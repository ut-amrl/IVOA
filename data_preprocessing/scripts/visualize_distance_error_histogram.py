from matplotlib import pyplot as plt
import random

f = open('dist_error_histogram.csv')

counts = []
boundaries = []

for line in f.readlines():
  nums = line.split(', ')
  lower = float(nums[0])
  upper = float(nums[1])
  boundaries.append(lower)
  count = int(nums[2])
  counts.append(count)

f.close()

window_f = open('dist_error_histogram_windows.csv')
windows = []
for line in window_f.readlines():
  nums = line.split(', ')
  pct = float(nums[0])
  pos = float(nums[1])
  neg = float(nums[2])
  color = (random.random(), random.random(), random.random())
  windows.append((pct, color, pos, neg))

window_f.close()

# add the last upper edge as well
boundaries.append(upper)

plt.hist(boundaries[:-1], weights=counts, bins=boundaries)
plt.xlabel('Error Estimation Distance')
plt.ylabel('Count')

for window in windows:
  # only label 1 so legend doesn't have duplicates
  plt.axvline(window[2], color=window[1], linestyle='dashed', linewidth=1, label=window[0])
  plt.axvline(window[3], color=window[1], linestyle='dashed', linewidth=1)
plt.legend()
plt.show()