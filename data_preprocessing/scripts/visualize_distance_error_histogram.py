from matplotlib import pyplot as plt
import random

HISTOGRAMS = [
  ('Error Distance', 'dist_error_histogram.csv', 'dist_error_histogram_windows.csv'),
  ('Relative Error Distance', 'rel_dist_error_histogram.csv', 'rel_dist_error_histogram_windows.csv'),

]

for name, histogram_file, window_file in HISTOGRAMS:
  f = open(histogram_file)
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

  window_f = open(window_file)
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
  plt.xlabel(name)
  plt.ylabel('Count')

  for window in windows:
    # only label 1 so legend doesn't have duplicates
    plt.axvline(window[2], color=window[1], linestyle='dashed', linewidth=1, label=window[0])
    plt.axvline(window[3], color=window[1], linestyle='dashed', linewidth=1)

  plt.legend()
  plt.show()