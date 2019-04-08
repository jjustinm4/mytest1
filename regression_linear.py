#we are trying to implement a linear regression model (exact theoretical implmentation)
import numpy as np
import random
import matplotlib.pyplot as plt 
#theta values are called regression coefficients initiate them to smaller random values
theta=[]
for c in range(2):
    theta.append(random.random())
#alpha is the learning rate , initiate it to a small value so the cost wont overshoot 
#and will possibly converge to 
#a global minimum    
alpha=.0001
#function to plot regression
def plot_regression_line(x, y,b): 
	plt.scatter(x, y, color = "m", 
			marker = "+", s = 300)  
	y_pred = b[0] + b[1]*x 
	plt.plot(x, y_pred, color= "b")
	plt.xlabel('x') 
	plt.ylabel('y') 
	plt.show()
 #function calculates the cost ,if cost is getting lower it means 
 #we are converging to some global optimum
def cost_function(X,Y,m):
    z=((theta[0]+(theta[1]*X))-Y)
    cost=.5*m*(sum(pow(z,2)))
    return cost
#this is the optimization part,this optimizes the coefficients 
#so tht they could give the best plot     
def gradient_descent(theta,X,Y,m):
    h_x=theta[0]+(theta[1]*X)
    temp0=theta[0]-((alpha/m)*(sum(h_x-Y)))
    temp1=theta[1]-((alpha/m)*(sum((h_x-Y)*X)))
    theta[0]=temp0
    theta[1]=temp1
def main():
#TEST DATA
    X=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    Y=np.array([2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40])
    c=[]
    m=len(X)
    cost_function(X,Y,m)
    #THE RANGE IS SOMETHING WE CAN DECIDE,IVE FOUND 1000000 AS AN OPTIMAL VALUE
    #SURE GREATER NUMBER OF ITERATIONS REDUCE COST BUT NOT SIGNIFICANTLY
    for i in range(1000000):
        #if our thetas are penalised as less than .0001 we assume it as an optimal value
        if cost_function(X,Y,m)<.0001:
            break
        gradient_descent(theta,X,Y,m)

    print("theta values aftr maximum iteraions are  ",theta[0]," ",theta[1])
    plot_regression_line(X, Y,theta) 

if __name__ == "__main__": 
	main()
         

    
    