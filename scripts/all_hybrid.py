f_1 = open('../output/hybrid_2.csv', 'r')
f_2 = open('../output/xgb_1.csv', 'r')
# f_3 = open('../output/output_xgboosting_8.csv', 'r')

l1 = f_1.read().splitlines()
l2 = f_2.read().splitlines()
# l3 = f_3.read().splitlines()


l1 = [line.split(',')[1] for line in l1]
l2 = [line.split(',')[1] for line in l2]
# l3 = [line.split(',')[1] for line in l3]

f_out = open('../output/hybrid_3.csv', 'w')
f_out.write("Id,Category\n")

for i in range(len(l1)):
    sumz = 0
    if i == 0:
        continue
    sumz += int(l1[i])
    sumz += int(l2[i])
    # sumz += int(l3[i])
    ans = 0
    if sumz > 0:
        ans = 1
    f_out.write(u"{0},{1}\n".format(i-1, ans))



