import sys
result=[]

input_file = str(sys.argv[1])
with open(input_file,'r') as f:
    for line in f:
        result = line.split()
s = []
for i in result:
    if i not in s:
        s.append(i)

counttt = []
for i in s:
    counttt.append(result.count(i))

txtlist = []
for ii in range(300):
    txtlist.append(s[ii] +' '+ str(ii) +' '+ str(counttt[ii]))

thefile = open('Q1.txt', 'w')
for item in txtlist:
    thefile.write(item)
    if txtlist.index(item) <299:
        thefile.write('\n')
thefile.close()