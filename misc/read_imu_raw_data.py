# this is a smaple script to read data
from matplotlib import pyplot as plt
import re
file_path= r'/Users/samhajhashemi/test.csv'
f=open(file_path,"r")
lines=f.readlines()
result=[]
for x in lines:
    try:
        result.append(float(re.sub('\n', '', x.split(' ')[7])))
        # or can do x.replace('\n','') instead
        # or can do x[:x.rfind("\n")]
        # or can do text.rsplit("\n",1)[0]
        # myString="Hello there !bob@"
        # mySubString=myString[myString.find("!")+1:myString.find("@")]
        # print(mySubString)
    except:
        print("no data")
f.close()
print(result)
plt.plot(result)
plt.show()

