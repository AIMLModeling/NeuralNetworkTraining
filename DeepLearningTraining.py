import numpy as np
import matplotlib.pyplot as plt
input_vector = np.array([10.66, 10.56])
target = 0.5
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])
learning_type=5
if learning_type ==1:
    learning_rate=0.05
elif learning_type ==2:
    learning_rate=0.01
elif learning_type ==3:
    learning_rate=0.15
elif learning_type ==4:
    learning_rate=0.2
else:
    learning_rate=0.1
critical_value=0.0001
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def make_prediction(input_vector, weights, bias):
     layer_1 = np.dot(input_vector, weights) + bias
     print(f"layer_1 is: {layer_1}")
     layer_2 = sigmoid(layer_1)
     print(f"layer_2 is: {layer_2}")
     return layer_2
print(f"weights_1 1st:{weights_1}")
prediction = make_prediction(input_vector, weights_1, bias)
print(f"The prediction result is: {prediction}")
error=(prediction-target)**2
print(f"Initial error is: {error}")
prediction_error=[]
step_number=1
while error > critical_value:
    mse = np.square(prediction - target)
    print(f"Prediction: {prediction}; Error: {mse}")
    derivative = 2 * (prediction - target)
    print(f"The derivative is {derivative}")
    adjustment=derivative*learning_rate
    print(f"The adjustment is {adjustment}")
    # Updating the weights
    print(f"Original weights_1 is: {weights_1}")
    weights_1 = weights_1 - adjustment
    print(f"Adjusted weights_1 is: {weights_1}")
    prediction = make_prediction(input_vector, weights_1, bias)
    difference=prediction - target
    error = difference ** 2
    prediction_error.append([prediction,error,step_number])
    print(f"Difference: {difference}; Step:{step_number}; error:{error}")
    step_number+=1
    if step_number>100:
        break
columns = list(zip(*prediction_error)) #transpose rows to columns
predictionlist=list(columns[0]) 
errorlist=list(columns[1])
indexlist=list(columns[2])
fig, ax = plt.subplots()
ax.scatter(predictionlist, errorlist)
for i, txt in enumerate(indexlist):
    ax.annotate(txt, (predictionlist[i], errorlist[i]))


