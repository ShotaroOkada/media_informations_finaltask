import cv2 as cv
import numpy as np

X = np.zeros((100, 8500), np.float32)
for i in range(100):
    filename = "LFW200/LFW" + str(i).zfill(3) + ".jpg"
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img[80:180, 80:165]
    img = img.reshape((1, 8500))
    X[i, :] = img[0, :]

mean = np.mean(X, axis=0)
X = X-mean

n = X.shape[0]
Cov = np.dot(X, X.T)/n
eigval, eigvec_v = np.linalg.eig(Cov)

eigvec_u = np.dot(X.T, eigvec_v)
for i in range(100):
    eigvec_u[:, i] /= np.linalg.norm(eigvec_u[:, i])

newimg = cv.imread("LFW200/LFW190.jpg")
newimg = cv.cvtColor(newimg, cv.COLOR_BGR2GRAY)
newimg = newimg[80:180, 80:165]
newimg = newimg.reshape((1, 8500))
newimg = list(map(np.float32, newimg))
newimg -= mean

dim_nums = [5, 10, 20, 30, 100]

for i in dim_nums:
    eigvec = eigvec_u[:, :i]
    PCAdim = np.dot(newimg, eigvec)
    revimg = np.dot(PCAdim, eigvec.T)
    revimg += mean
    cv.imwrite('dim' + str(i).zfill(3) + '.jpg', revimg.reshape((100, 85)))
