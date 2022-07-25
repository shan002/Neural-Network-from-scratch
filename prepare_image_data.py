import numpy as np
import glob
import cv2

WIDTH = 64
HEIGHT = 64

def save_image_data():
    hotdogs = []
    for img_name in glob.glob('hot_dog/' + '*.jpg'):
        img = cv2.imread(img_name)
        img_resized = cv2.resize(img, (WIDTH, HEIGHT))
        hotdogs.append(img_resized)
        # cv2.imshow('pre', img)
        # cv2.imshow('post', img_resized)
        # cv2.waitKey(0)

    nothotdogs = []
    for img_name in glob.glob('not_hot_dog/' + '*.jpg'):
        img = cv2.imread(img_name)
        img_resized = cv2.resize(img, (WIDTH, HEIGHT))
        nothotdogs.append(img_resized)
        # cv2.imshow('pre', img)
        # cv2.imshow('post', img_resized)
        # cv2.waitKey(0)

    hotdogs = np.array(hotdogs, dtype='uint8')
    nothotdogs = np.array(nothotdogs, dtype='uint8')
    print(hotdogs.shape, nothotdogs.shape)
    hotdogs = hotdogs.reshape(hotdogs.shape[0], -1).T
    nothotdogs = nothotdogs.reshape(nothotdogs.shape[0], -1).T
    print(hotdogs.shape, nothotdogs.shape)
    X = np.concatenate((hotdogs, nothotdogs), axis=1)
    y = np.concatenate((np.ones((1, hotdogs.shape[1]), dtype='uint8'), np.zeros((1, nothotdogs.shape[1]), dtype='uint8')), axis = 1)

    # for i in range(X.shape[1]):
    #     cv2.imshow('image_resized', X[:, i].reshape(WIDTH, HEIGHT, 3))
    #     cv2.waitKey(0)

    with open('image_data.npy', 'wb') as f:
        np.save(f, X)
        np.save(f, y)

    print("Data saved!")

def load_image_data():
    with open('image_data.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)

    return X, y


# save_image_data()

# X, y = load_image_data()
# print(X.dtype, X.shape)
# print(y.dtype, y.shape)

# print(y)
# for i in range(X.shape[1]):
#     cv2.imshow('image_resized', X[:, i].reshape(WIDTH, HEIGHT, 3))
#     cv2.waitKey(0)
