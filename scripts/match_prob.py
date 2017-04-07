f_1 = open('../output/gold.csv', 'r')
f_2 = open('../output/xgb_5.csv', 'r')
probs_file = open('../output/xgb_5_prob.csv','r')
# f_3 = open('../output/output_xgboosting_8.csv', 'r')

l1 = f_1.read().splitlines()
l2 = f_2.read().splitlines()
probs = probs_file.read().splitlines()
# l3 = f_3.read().splitlines()


l1 = [line.split(',')[1] for line in l1]
l2 = [line.split(',')[1] for line in l2]
probs[0] = '1 2'
left = [float(line.split(' ')[0]) for line in probs]
right = [float(line.split(' ')[1]) for line in probs]
# l3 = [line.split(',')[1] for line in l3]

f_out = open('../output/hybrid_5.csv', 'w')
f_out.write("Id,Category\n")

for i in range(len(l1)):
    if i == 0:
        continue
    ans = 0
    if l1[i] == l2[i]:
        ans = l1[i]

    elif l1[i] == '1':
        if left[i] > 0.78:
            ans = 0
        else:
            ans = 1

    elif l1[i] == '0':
        if right[i] > 0.78:
            ans = 1
        else:
            ans = 0

    f_out.write(u"{0},{1}\n".format(i-1, ans))



