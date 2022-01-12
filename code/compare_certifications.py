import argparse
from decimal import *

parser = argparse.ArgumentParser(description='Compare certification results')
parser.add_argument("file0", type=str)
parser.add_argument("file1", type=str)
args = parser.parse_args()

with open(args.file0) as inp:
    next(inp)
    rows = [line.strip().split('\t') for line in inp]
    if rows[-1][0].startswith("radius sum"):
        rows.pop()
    columns0 = list(zip(*rows))
with open(args.file1) as inp:
    next(inp)
    rows = [line.strip().split('\t') for line in inp]
    if rows[-1][0].startswith("radius sum"):
        rows.pop()
    columns1 = list(zip(*rows))

limit = min(len(columns0[0]), len(columns1[0]))
getcontext().prec = 32 * 32 * 3 * 2

correct0 = 0
correct1 = 0
correct_both = 0
relative_radius_sum = 0.
base_radius_sum = 0.
certified_volume0 = Decimal(0.)
certified_volume1 = Decimal(0.)
certified_radius_sum0 = 0.
certified_radius_sum1 = 0.
for i in range(limit):
    if columns0[4][i] == '1':
        correct0 += 1
        certified_volume0 += Decimal(float(columns0[3][i])) ** Decimal(32 * 32 * 3)
        certified_radius_sum0 += float(columns0[3][i])
    if columns1[4][i] == '1':
        correct1 += 1
        certified_volume1 += Decimal(float(columns1[3][i])) ** Decimal(32 * 32 * 3)
        certified_radius_sum1 += float(columns1[3][i])
    if columns0[4][i] == '1' and columns1[4][i] == '1':
        correct_both += 1
        relative_radius = float(columns1[3][i]) - float(columns0[3][i])
        base_radius_sum += float(columns0[3][i])
        # print(relative_radius)
        relative_radius_sum += relative_radius

print('all statistics are for the images which were (tried to be) certified in both files')
print('correctly classified images in file 0:', correct0)
print('correctly classified images in file 1:', correct1)
print('correctly classified images in both files:', correct_both)
print('mean increase in certified radius:', relative_radius_sum / correct_both)
print('change in correctly classified images: {:.3}%'.format((correct1 / correct0 - 1) * 100))
print('change in certified radius: {:.3}%'.format(((relative_radius_sum + base_radius_sum) / base_radius_sum - 1) * 100))
print('change in certified radius sum: {:.3}%'.format((certified_radius_sum1 / certified_radius_sum0 - 1) * 100))
# print('certified volume in file 0:', certified_volume0)
# print('certified volume in file 1:', certified_volume1)
# print('relative certified volume:', certified_volume1 / certified_volume0)
