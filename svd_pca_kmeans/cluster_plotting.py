import matplotlib.pyplot as plt
from scipy.spatial import distance

centroids = []
X = [[[], []], [[], []], [[], []]]
Cent = [[], []]
filepath = '/Users/ruhulislam/PycharmProjects/pythonProject/KMCentroid_plot.txt'
with open(filepath) as fp:
    line = fp.readline()
    while line:
        if line:
            line = line.strip()
            cord = line.split(',')
            centroids.append((float(cord[0]), float(cord[1])))
            Cent[0].append((float(cord[0])))
            Cent[1].append((float(cord[1])))
        else:
            break
        line = fp.readline()
fp.close()
list1 = []
filepath = '/Users/ruhulislam/PycharmProjects/pythonProject/KMclusterd_output_plot.txt'
with open(filepath) as fp:
    line = fp.readline()
    while line:
        if line:
            line = line.strip()
            list1 = line.split(',')
            x = (float(list1[2]), float(list1[3]))
            dist = 100000000000000
            selected_m = -1
            for m in centroids:
                test_distance = distance.euclidean(x, m)
                if test_distance < dist:
                    dist = test_distance
                    selected_m = centroids.index(m)
            X[selected_m][0].append(x[0])
            X[selected_m][1].append(x[1])
        else:
            break
        line = fp.readline()
fp.close()
plt.plot(X[0][0], X[0][1], 'ro')
plt.plot(X[1][0], X[1][1], 'go')
plt.plot(X[2][0], X[2][1], 'bo')
plt.plot(Cent[0], Cent[1], 'yo')  # centroids are yellow color
plt.show()
