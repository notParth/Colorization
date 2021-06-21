import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rand
import math

# creates the 3*3 patch vector for a given pixel
def neighbors(pos, img):
    i, j = pos
    n = img.shape[0]
    m = img.shape[1]
    neighbors = []

    neighbors.append(img[i][j])
    if i > 0:
        neighbors.append(img[i-1][j])
    else:
        neighbors.append(img[i][j])
    if i < n-1:
        neighbors.append(img[i+1][j])
    else:
        neighbors.append(img[i][j])
    if j > 0:
        neighbors.append(img[i][j-1])
    else:
        neighbors.append(img[i][j])
    if j < m-1:
        neighbors.append(img[i][j+1])
    else:
        neighbors.append(img[i][j])
    if i > 0 and j > 0:
        neighbors.append(img[i-1][j-1])
    else:
        neighbors.append(img[i][j])
    if i < n-1 and j < m-1:
        neighbors.append(img[i+1][j+1])
    else:
        neighbors.append(img[i][j])
    if i > 0 and j < m-1:
        neighbors.append(img[i-1][j+1])
    else:
        neighbors.append(img[i][j])
    if i < n-1 and j > 0:
        neighbors.append(img[i+1][j-1])
    else:
        neighbors.append(img[i][j])
 
    return np.array(neighbors)

# creates the quadratic features of a given data point
def make_features(points):
    points = list(points)
    preprocessed = [1.0]
    length = len(points)
    for i in range(length):
        preprocessed.append(points[i])
        for j in range(i, length):
            preprocessed.append(points[i]*points[j])
    return np.array(preprocessed)

# sigmoid function
def sigmoid(x):
    return (1/(1+np.exp(-x)))

# model
# component: 0 -> red, 1 -> green, 2 -> blue
# alpha = learning rate
def model(input, output, component, alpha):
    # random weights between -0.001 and 0.001
    weights = np.random.uniform(-0.001, 0.001, 55)
    # preprocessing the output
    output = output[:, component] / 255
    while(True):
        # best result at 0.01
        random_index = rand.randint(0, model_input.shape[0]-1)
        random_point = input[random_index]
        value = sigmoid(np.matmul(random_point, weights))
        gradient = alpha * 2 * (value - output[random_index]) * value * (1 - value) * random_point
        weights = weights - gradient

        error = math.sqrt(((np.apply_along_axis(sigmoid, 0, np.matmul(input, weights)) - output)**2).mean())
        print(error)
        if component == 0:
            if error < 0.13: #0.1193, 0.0693, 0.05347
                return weights
        elif component == 1:
            if error < 0.05: #0.0338, 0.0209, 0.015, 0.01765
                return weights
        else:
            if error < 0.058: #0.0525, 0.172, 0.103
                return weights

# load the image
img = mpimg.imread('C:/Users/Parth/Desktop/RU/CS440/project_four/sample_images/sa.jpg')
# creating the grayscale image
gray_img = np.matmul(img, np.transpose(np.array([0.21, 0.72, 0.07])))
# preprocessing: normalizing over 255
gray_img = gray_img / 255
# splitting into training and testing data
img_train, img_test = np.hsplit(img, 2)
gray_train, gray_test = np.hsplit(gray_img, 2)

# preprocessing: recentering and features
input_mean = gray_train.mean()
model_input = np.array([[neighbors((i, j), gray_train) for j in range(1, gray_train.shape[1]-1)] for i in range(1, gray_train.shape[0]-1)])
model_input = model_input.reshape(model_input.shape[0]*model_input.shape[1], 9)
model_input = model_input - input_mean
model_input = np.array([make_features(i) for i in model_input])

# output
model_output = np.array([[img_train[i][j] for j in range(1, img_train.shape[1]-1)] for i in range(1, img_train.shape[0]-1)])
model_output = model_output.reshape(model_output.shape[0]*model_output.shape[1], 3)

# red model
print('Figuring out the red model')
red_weights = model(model_input, model_output, 0, 0.1)

# green model
print('Figuring out the green model')
green_weights = model(model_input, model_output, 1, 0.1)

# blue model
print('Figuring out the blue model')
blue_weights = model(model_input, model_output, 2, 0.1)

print(red_weights)
print(green_weights)
print(blue_weights)

# copy of the image, since original is not modifiable
new_img = img.copy()
img_train_two , img_test_two = np.hsplit(new_img, 2)

red_error = []
green_error = []
blue_error = []
# intelligent stuff ahead
# beep boop
input_mean = gray_test.mean()
gray_test = gray_test - input_mean
for i in range(gray_test.shape[0]):
    for j in range(gray_test.shape[1]):
        red_value = 255 * sigmoid(np.matmul(red_weights, make_features(neighbors((i, j), gray_test))))
        green_value = 255 * sigmoid(np.matmul(green_weights, make_features(neighbors((i, j), gray_test))))
        blue_value = 255 * sigmoid(np.matmul(blue_weights, make_features(neighbors((i, j), gray_test))))
        img_test_two[i][j] = np.array([red_value, green_value, blue_value])
        red_error.append((red_value - img_test[i][j][0])**2)
        green_error.append((green_value - img_test[i][j][1])**2)
        blue_error.append((blue_value - img_test[i][j][2])**2)

print(math.sqrt(np.mean(red_error))) # 27.14
print(math.sqrt(np.mean(blue_error))) # 29.83
print(math.sqrt(np.mean(green_error))) # 13.19
imgplot = plt.imshow(new_img)
plt.show()