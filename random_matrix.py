# 합이 1인 난수 행렬
import numpy as np 
column = 3
raw = 3   
reference = np.random.uniform(low=0, high=1.0, size=(column,raw - 1)) 

reference.sort(axis = 1) 

diffs = np.diff(reference, prepend=0, append=1, axis=1) 

print(diffs) 
print()
print(diffs[:,0])
print()
print(np.sum(diffs, axis=1))