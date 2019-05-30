# polarity data에 새로운 데이터 셋을 추가하는 코드

import os

f= open("rt-polaritydata/rt-polarity-Copy1.neg","+a")
count = 0

for root, dirs, files in os.walk('newdata/pos'):
    for fname in files:
        
        file_name = os.path.join(root, fname)
        count += 1
        f2=open(file_name, "r")
        
        #Open the file back and read the contents
        if f2.mode == 'r':
            contents =f2.read()
            f.write(contents+"\n")
            
print(count)
                        
f.close()
f2.close()