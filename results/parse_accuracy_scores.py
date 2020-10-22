import re
import pandas as pd

f=open("ecl.txt", "r")
output_filename = 'ecl_result.csv'
text = f.read()

x = re.findall(" [0-9]*/[0-9]* ", text)
print(x)



count = 1
scores = []
for i in x:
    if count >1 and count % 14 == 0:
        scores.append(int(i[0:-6])/2946 * 100)
    count += 1

result = pd.DataFrame(scores)
print(result)
result.T.to_csv(output_filename, encoding='utf-8', index=False, header=None)
