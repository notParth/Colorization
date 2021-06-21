import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rand
import sys
import math

# kmeans clustering algorithm
# takes image and the number of clusters as input
def kmeans(img, n=5):
    cnt = 0
    shape_x = img.shape[0]
    shape_y = img.shape[1]
    coordinates = []
    # randomly choose 'n' cluster averages
    while (cnt < n):
        random_x = rand.randint(0, shape_x-1)
        random_y = rand.randint(0, shape_y-1)
        if (random_x, random_y) in coordinates:
            continue
        coordinates.append((random_x, random_y))
        cnt+=1
    clusters_avg = np.array([img[i[0]][i[1]] for i in coordinates])
    flag = True
    # start sorting the dataset according to the cluster averages
    while(flag):
        temp = np.zeros([1, 3])
        temp[:] = np.nan
        clusters = [temp for x in range(n)]
        for i in img:
            for j in i:
                min = sys.maxsize
                belongs_in = -1
                # see which cluster average is the closest to the a given datapoint
                # np.linalg.norm() is used as the euclidean distance
                for index, k in enumerate(clusters_avg):
                    dist = np.linalg.norm(j-k)
                    if min > dist:
                        min = dist
                        belongs_in = index
                clusters[belongs_in] = np.append(clusters[belongs_in], [j], axis=0)
        new_avg_cluster = []
        # calculate the new cluster averages
        for i in range(n):
            new_avg_cluster.append(np.nanmean(clusters[i], axis=0))
        new_avg_cluster = np.array(new_avg_cluster)
        # terminating condition
        # terminates if the euclidean distance between the new cluster avgs and the old cluster avgs is less than 0.0000001 
        if np.linalg.norm(new_avg_cluster-clusters_avg) < 0.0000001:
            flag = False
            clusters_avg = new_avg_cluster
        clusters_avg = new_avg_cluster
    return clusters_avg

# creates the 3*3 patch vector for a given pixel
def neighbors(pos, img):
    i, j = pos
    n = img.shape[0]
    m = img.shape[1]
    neighbors = []

    neighbors.append(img[i][j])
    if i > 0:
        neighbors.append(img[i-1][j])
    if i < n-1:
        neighbors.append(img[i+1][j])
    if j > 0:
        neighbors.append(img[i][j-1])
    if j < m-1:
        neighbors.append(img[i][j+1])
    if i > 0 and j > 0:
        neighbors.append(img[i-1][j-1])
    if i < n-1 and j < m-1:
        neighbors.append(img[i+1][j+1])
    if i > 0 and j < m-1:
        neighbors.append(img[i-1][j+1])
    if i < n-1 and j > 0:
        neighbors.append(img[i+1][j-1])
 
    return np.array(neighbors)

# finds the six most identical patches for a given 'patch' in image(training)
# returns the indices of the patches
def find_patch(patch, img):
    patches_dist = []
    patches_index = []
    # compares each patch in the training image with the given 'patch'
    # patches_index stores the indices while patches_dist stores the distance for each index
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            current_patch = neighbors((i, j), img)
            distance = np.linalg.norm(np.linalg.norm(current_patch-patch))
            patches_dist.append(distance)
            patches_index.append((i, j))
    patches_dist = np.array(patches_dist)
    
    # sorts the distance matrix and find the indices of the 6 smallest distances
    dist_closest_indices = np.argsort(patches_dist)[:6]
    # gets the indices of the pixel based on the smallest distance
    closest_indices = [patches_index[i] for i in dist_closest_indices]
    return closest_indices

# basic agent
# grayscale = grayscale image, five_color = five color representative image, five_colors = (r, g, b) of the five representative colors
def basic_agent(grayscale, five_color, five_colors):
    color_training_data, color_testing_data = np.hsplit(five_color, 2)
    gray_training_data, gray_testing_data = np.hsplit(grayscale, 2)
    for i in range(1, gray_testing_data.shape[0]-1):
        for j in range(1, gray_testing_data.shape[1]-1):
            identical = find_patch(neighbors((i, j), gray_testing_data), gray_training_data)
            colors = np.array([color_training_data[m][n] for m, n in identical])

            # decides which representative color to choose
            five_colors_cnt=[0, 0, 0, 0, 0]
            for k in colors:
                if (k==five_colors[0]).all():
                    five_colors_cnt[0] += 1
                if (k==five_colors[1]).all():
                    five_colors_cnt[1] += 1
                if (k==five_colors[2]).all():
                    five_colors_cnt[2] += 1
                if (k==five_colors[3]).all():
                    five_colors_cnt[3] += 1
                if (k==five_colors[4]).all():
                    five_colors_cnt[4] += 1
            five_colors_cnt = np.array(five_colors_cnt)
            temp = np.argsort(five_colors_cnt)
            if five_colors_cnt[temp[-1]] == five_colors_cnt[temp[-2]]:
                # chooses the closest resembling color
                color_testing_data[i][j] = colors[0]
            else:
                # chooses the majority color
                color_testing_data[i][j] = five_color[temp[-1]]
    
    return five_color

# loading the image
img = mpimg.imread('C:/Users/Parth/Desktop/RU/CS440/project_four/sample_images/sample_three.jpg')
imgplot = plt.imshow(img)
plt.show()

# figuring the five representative colors
five_colors = kmeans(img)

five_color_img = np.zeros_like(img)
gray_img = np.zeros_like(img)

# creating the grayscale and the five color representative images
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        min = sys.maxsize
        replacement = np.nan
        for k in five_colors:
            this_norm = np.linalg.norm(img[i][j]-k)
            if min > this_norm:
                min = this_norm
                replacement = k
        five_color_img[i][j] = replacement
        grayscale = np.sum(img[i][j] * np.array([0.21, 0.72, 0.07]))
        gray_img[i][j] = np.array([grayscale, grayscale, grayscale])


# blackening the border of the testing part
first_half, second_half = np.hsplit(five_color_img, 2)
# copy for later
org = np.copy(second_half)
for i in range(second_half.shape[0]):
    for j in range(second_half.shape[1]):
        if i==0 or j==0 or i==second_half.shape[0]-1 or j==second_half.shape[1]-1:
            second_half[i][j] = np.array([0, 0, 0])

intelligent = basic_agent(gray_img, five_color_img, five_colors)
_, gen = np.hsplit(intelligent, 2)
# calculating the mathematical difference in the original five colors image and the agent generated
red_error = []
blue_error = []
green_error = []
cnt = 0
for i in range(org.shape[0]):
    for j in range(org.shape[1]):
        if not np.array_equal(org[i][j], gen[i][j]):
            cnt += 1
        red_error.append((org[i][j][0] - gen[i][j][0])**2)
        green_error.append((org[i][j][1] - gen[i][j][1])**2)
        blue_error.append((org[i][j][2] - gen[i][j][2])**2)

print(cnt) # 6025
print(math.sqrt(np.mean(red_error))) # 96.2517064406193
print(math.sqrt(np.mean(green_error))) # 104.08532505830132
print(math.sqrt(np.mean(blue_error))) # 102.80631356131244
imgplot = plt.imshow(intelligent)
plt.show()
