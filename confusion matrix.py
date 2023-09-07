import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

a = np.array(['dog','not dog','dog','not dog','dog'])
b = np.array(['dog','dog','not dog','dog','dog'])

cm = confusion_matrix(a,b)
print(cm)
sns.heatmap(cm,
            cmap='Reds',
            annot=True,
            fmt='g',
            xticklabels=["dog","not dog"],
            yticklabels=["dog","not dog"])
plt.xlabel="actual"
plt.ylabel="predicted"
plt.show()
