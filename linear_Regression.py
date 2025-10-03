import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("test.csv")

data.rename(columns = {'x': "input", 'y': 'output'}, inplace = True)

# Mean square Erroe
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].input
        y = points.iloc[i].output
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

# Gradient Descent
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].input
        y = points.iloc[i].output
    
        m_gradient += - (2/n) * x * (y - (m_now * x + b_now))
        b_gradient += - (2/n) * (y - (m_now * x + b_now))
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

#Training
m = 0
b = 0
L = 0.0001
epochs = 1000

for i in range (epochs) :
    if i % 50 == 0:
        loss = loss_function(m, b, data)
        print(f"Epoch {i}: m = {m:.4f} b = {b:.4f} loss = {loss:.4f}")
    m, b = gradient_descent(m, b, data, L)
    
# Final parameters
print(f"\nFinal parameters: m={m:.4f}, b={b:.4f}, loss={loss_function(m, b, data):.4f}")

# Plotting results

plt.scatter(data.input, data.output, color="black", label = 'Data')
# Creating x values for the line
x_vals = np.linspace(data.input.min(), data.input.max(), 100)

# Predicting value for y
y_vals = m * x_vals +b

plt.plot(x_vals, y_vals, color ='red', label = 'Best fit Line')
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.show()