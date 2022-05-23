#%matplotlib inline
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


from sklearn.neighbors import NearestNeighbors
filenames = pickle.load(open('filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(open('features-caltech101-resnet.pickle', 'rb'))
#print(feature_list)

neighbors=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean').fit(feature_list)
distances,indices=neighbors.kneighbors([feature_list[0]])
'''print(neighbors)
print(distances)
print(indices[0])
print(filenames[0])'''
#see the actual image behind feature, image to query in index 0
plt.imshow(mpimg.imread(filenames[0]))
#plt.imshow(mpimg.imread(filenames[indices[1]])) #errrrooorr
for i in range(5):
    print(distances[0][i])#should print only 4 values and the first one should be 0 correspondign with the distance from itself, doesnt work
plt.show()

for i in range(6):
    random_image_index = random.randint(0,9144)#number of images in the databse
    distances, indices = neighbors.kneighbors([feature_list[random_image_index]])
    # don't take the first closest image as it will be the same image
    similar_image_paths = [filenames[random_image_index]] + [filenames[indices[0][i]] for i in range(1,4)]
    plot_images(similar_image_paths, distances[0])#helper function that visualizes several query images with their nearest neighbors