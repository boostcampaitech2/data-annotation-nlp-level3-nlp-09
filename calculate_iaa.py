import pandas as pd
import numpy as np
from fleiss import fleissKappa

result = pd.read_excel('q100_3.xlsx',engine='openpyxl')
result = result.to_numpy()
num_classes = int(np.max(result)) + 1

transformed_result = []
for i in range(len(result)):
    temp = np.zeros(num_classes)
    for j in range(len(result[i])):
        temp[int(result[i][j])] += 1
    transformed_result.append(temp.astype(int).tolist())

kappa = fleissKappa(transformed_result,len(result[0]))