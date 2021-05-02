import re
import os

# read input from R to Python
# save_path = '/Users/ruhulislam/PycharmProjects/pythonProject/testdata.txt'
save_path = '/Users/ruhulislam/PycharmProjects/pythonProject/outpcaZWithIDCLID.txt'

f = open(save_path, 'r')
lines = f.readlines()

# f2 = open('/Users/ruhulislam/PycharmProjects/pythonProject/KMTestData_by_testdata.txt', 'a')
f2 = open('/Users/ruhulislam/PycharmProjects/pythonProject/KMTestData_by_outpcaZWithIDCLID.txt', 'a')
i = 1
# clean data to formatted data
for line in lines:
    re_key1 = re.compile(' ')
    #re_key2 = re.compile("'")
    re_key5 = re.compile("\n")
    #re_key3 = re.compile(" ")
    line = re_key5.sub('',line)
    a = re_key1.split(line)
    if i <= 9:
        a[0] = a[0][a[0].find('.') - 1:a[0].find('.')]
        #print(a[0])
    if i > 9 and i <= 99:
        ID1 = a[0][a[0].find('.') - 1:a[0].find('.')]
        ID2 = a[0][a[0].find('.') + 1:a[0].find('.') + 2]
        a[0] = ID1 + ID2
        #print(a[0])
    if i > 99 and i <= 190:
        ID1 = a[0][a[0].find('.') - 1:a[0].find('.')]
        ID2 = a[0][a[0].find('.') + 1:a[0].find('.') + 3]
        a[0] = ID1 + ID2
    #print(a)
    # b = re_key2.sub('', b)
    # b = re_key3.sub('',b)
    # output = a[0] + '|' + a[3] + '|' + a[1] +','+ a[2] +'\n'
    # output = a[0] + '|' + a[1] + '|' + a[2] + ',' + a[3] + ',' + a[4] + ',' + a[5] + ',' + a[6] +'\n'
    # output = a[0] + '|' + a[1][a[1].find('.')-1:a[1].find('.')] + '|' + a[2] + ',' + a[3] + ',' + a[4] + '\n'
    output = a[0] + '|' + a[1][a[1].find('.') - 1:a[1].find('.')] + '|' + a[2] + ',' + a[3] + '\n'
    i += 1
    f2.write(output)

f2.close()
f.close()
