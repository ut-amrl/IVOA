from matplotlib import pyplot as plt

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

# add the last upper edge as well
boundaries.append(upper)

plt.hist(boundaries[:-1], weights=counts, bins=boundaries)
plt.xlabel('Error Estimation Distance')
plt.ylabel('Count')
plt.show()