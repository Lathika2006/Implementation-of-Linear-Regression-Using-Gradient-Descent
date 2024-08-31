# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:

```

Program to implement the linear regression using gradient descent.
Developed by: LATHIKA L J
RegisterNumber:  212223220050

import numpy as  np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=100):
  X=np.c_[np.ones(len(X1)),x1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)

  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)        
    theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta

data=pd.read_csv("50_Startups.csv")
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled);

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:
![Screenshot 2024-08-31 140603](https://github.com/user-attachments/assets/5d8fadbb-67e8-4c39-b4b9-bf74b3b70e19)
![Screenshot 2024-08-31 140555](https://github.com/user-attachments/assets/7fbf6c22-4c70-4239-923d-2b95fd371c35)
![Screenshot 2024-08-31 140546](https://github.com/user-attachments/assets/4e73de91-da8d-4a5c-8105-f852e5846ecb)
![Screenshot 2024-08-31 140535](https://github.com/user-attachments/assets/0a27a7d8-754e-4d4a-be80-b02967021020)
![Screenshot 2024-08-31 140526](https://github.com/user-attachments/assets/bbdf1a50-e1f6-4670-8a7f-e1d30ef18e0e)
![Screenshot 2024-08-31 140516](https://github.com/user-attachments/assets/97b3384b-a6e1-4940-a884-a1223194557a)
![Screenshot 2024-08-31 140501](https://github.com/user-attachments/assets/82cf341a-f0c2-4269-b91e-4a3ed706abe6)
![Screenshot 2024-08-31 140449](https://github.com/user-attachments/assets/04f9ac50-0540-402c-ba8a-6494a5671681)
![Screenshot 2024-08-31 140440](https://github.com/user-attachments/assets/eea2e4c2-8365-4336-ae2d-6f2dcda5c9c9)
![Screenshot 2024-08-31 140428](https://github.com/user-attachments/assets/3e375528-c939-4686-8d2c-092afd6ce1e8)
![Screenshot 2024-08-31 140412](https://github.com/user-attachments/assets/c25a9ce5-736e-43c8-8f47-646223710282)
![Screenshot 2024-08-31 140350](https://github.com/user-attachments/assets/18385998-acfe-4b9f-a34a-6db9f81e7c00)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
