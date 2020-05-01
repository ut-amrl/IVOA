from matplotlib import pyplot as plt
import random
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--dir', type=str, required=True, help='Directory to evaluation results')
parser.add_argument('--plot_windows', action='store_true')
args = parser.parse_args()

HISTOGRAMS = [
  ('Error Distance', 'dist_error_histogram.csv', 'dist_error_histogram_windows.csv'),
  ('Relative Error Distance', 'rel_dist_error_histogram.csv', 'rel_dist_error_histogram_windows.csv'),
  ('Error Track Length', 'error_track_size_histogram.csv', None),
]

for name, histogram_file, window_file in HISTOGRAMS:
  f = open(os.path.join(args.dir, histogram_file))
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

  windows = []
  if window_file is not None:
    window_f = open(os.path.join(args.dir, window_file))
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

  if args.plot_windows:
    for window in windows:
      # only label 1 so legend doesn't have duplicates
      plt.axvline(window[2], color=window[1], linestyle='dashed', linewidth=1, label=window[0])
      plt.axvline(window[3], color=window[1], linestyle='dashed', linewidth=1)

  plt.legend()
  plt.show()