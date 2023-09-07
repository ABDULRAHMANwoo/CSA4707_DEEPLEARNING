import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
np.random.seed(0)
x=np.linspace(0,10,100).reshape(-1,1)
m=2
c=1
y=m*x+c+np.random.randn(100,1)

mo = LinearRegression()
mo.fit(x,y)
plt.plot(x,mo.predict(x))
plt.scatter(x,y)
plt.show()
