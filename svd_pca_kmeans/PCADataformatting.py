import re
import os

# read input from R to Python
save_path = '/Users/ruhulislam/PycharmProjects/pythonProject/pcaorgdata.txt'
print(save_path)
f = open(save_path, 'r')
lines = f.readlines()
f2 = open('/Users/ruhulislam/PycharmProjects/pythonProject/pcaorg_formatted.txt', 'a')
i = 1
# clean data to formatted data
for line in lines:
    re_key1 = re.compile(' ')
    # re_key2 = re.compile("'")
    re_key5 = re.compile("\n")
    # re_key3 = re.compile(" ")
    line = re_key5.sub('', line)
    a = re_key1.split(line)
    #print(a)
    # b = re_key2.sub('', b)
    # b = re_key3.sub('',b)
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
        #print(a[0])
    # output = a[0] + '|' + a[3] + '|' + a[1] +','+ a[2] +'\n'
    # output = a[0] + '|' + a[1] + '|' + a[2] + ',' + a[3] + ',' + a[4] + ',' + a[5] + ',' + a[6] +'\n'
    output = a[0] + '|' + a[1] + ',' + a[2] + ',' + a[3] + ',' + a[4] + ',' + a[5] + ',' + a[6] + ',' + \
             a[7] + ',' + a[8] + ',' + a[9] + ',' + a[10] + ',' + a[11] + ',' + a[12] + ',' +\
             a[13] + ',' + a[14] + ',' + a[15] + ',' + a[16] + ',' + a[17] + ',' + a[18] + ',' +\
             a[19] + ',' + a[20] + ',' + a[21] + ',' + a[22] + ',' + a[23] + ',' + a[24] + ',' +\
             a[25] + ',' + a[26] + ',' + a[27] + ',' + a[28] + ',' + a[29] + ',' + a[30] + ',' +\
             a[31] + ',' + a[32] + ',' + a[33] + ',' + a[34] + ',' + a[35] + ',' + a[36] + ',' +\
             a[37] + '\n'
    i += 1
    #     a[a.find('.')-1:a.find('.')]
    #     a[1][a[1].find('.')-1:a[1].find('.')]
    f2.write(output)
#
f2.close()
f.close()
