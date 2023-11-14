
### Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your compiler
2. Type the required program
3. Print the program.
4. End the program.

## Program:
```python
/*
Program to implement the linear regression using gradient descent.
Developed by: Vikash A R
RegisterNumber:  212222040179
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population od City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  
  m=len(y)
  h=X.dot(theta)
  square_err=(h - y)**2
  
  return 1/(2*m) * np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
print("Compute Cost Value")
computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):

  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = X.dot(theta) 
    error = np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta , J_history
 
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) Value")
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def predict(x,theta):

  predictions= np.dot(theta.transpose(),x)

  return predictions[0]
 
predict1=predict(np.array([1,3.5]),theta)*10000
print("Profit for the Population 35,000")
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("Profit for the Population 70,000")
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:

Profit Prediction graph

![269251778-6fe4d91a-f4cc-4eb0-8a29-d417cacf1e7d](https://github.com/VIKASHAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119405655/7b9915c5-0e51-45f4-bde7-5b9276dd20fc)

Compute Cost value

![274933197-ea631b45-5905-4e32-a2f2-5038df301277](https://github.com/VIKASHAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119405655/1cb9d38c-55eb-4ef2-b4df-ce5e3bb9c45f)

h(x) value

![274933421-a6355f1a-73f9-4a23-b2d2-268600f52b65](https://github.com/VIKASHAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119405655/9b558f49-6bfe-40ba-a28c-0f0e9d405354)

Cost function using Gradient descent graph

![274933481-9965d247-e8f6-4269-b0d5-c9fa362b2eb0](https://github.com/VIKASHAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119405655/f258771c-18ef-4e95-b253-65224f1928e3)

Profit Prediction graph

![274934171-a3f442ac-4285-44a8-962c-7f18b1a099f3](https://github.com/VIKASHAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119405655/29ed758a-c8b2-4472-b7ef-731cbc1cc995)

Profit for the population of 35000

![274933349-1c626390-aa3d-422c-afe8-e9819543e202](https://github.com/VIKASHAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119405655/b2dd8da4-36ea-4da1-8c53-7fdfce8f27fa)

Profit for the population of 70000

![274934255-c0fb798f-74a5-4ffb-9fb4-47256bba91db](https://github.com/VIKASHAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119405655/313229f2-2eb7-4c24-a4e4-b2bf5aeb0a0b)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
