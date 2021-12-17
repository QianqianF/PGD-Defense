import argparse

parser = argparse.ArgumentParser(description='Compare certification results')
parser.add_argument("file0", type=str)
parser.add_argument("file1", type=str)
args = parser.parse_args()

with open(args.file0) as inp:
    next(inp)
    columns0 = list(zip(*(line.strip().split('\t') for line in inp)))
with open(args.file1) as inp:
    next(inp)
    columns1 = list(zip(*(line.strip().split('\t') for line in inp)))

limit = min(len(columns0[0]), len(columns1[0]))

correct0 = 0
correct1 = 0
correct_both = 0
relative_radius_sum = 0.
certified_volume0 = 0.
certified_volume1 = 0.
for i in range(limit):
    if columns0[4][i] == '1':
        correct0 += 1
        certified_volume0 += float(columns0[3][i]) ** (32 * 32 * 3)
    if columns1[4][i] == '1':
        correct1 += 1
        certified_volume1 += float(columns1[3][i]) ** (32 * 32 * 3)
    if columns0[4][i] == '1' and columns1[4][i] == '1':
        correct_both += 1
        relative_radius = float(columns1[3][i]) / float(columns0[3][i])
        print(relative_radius)
        relative_radius_sum += relative_radius

print('all statistics are for the images which were (tried to be) certified in both files')
print('correctly classified images in file 0:', correct0)
print('correctly classified images in file 1:', correct1)
print('correctly classified images in both files:', correct_both)
print('mean relative increase in certified radius:', relative_radius_sum / correct_both)
print('certified volume in file 0:', certified_volume0)
print('certified volume in file 1:', certified_volume1)
print('relative certified volume:', certified_volume1 / certified_volume0)
