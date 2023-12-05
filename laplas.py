import cv2
import numpy as np
import utils


def laplassian_method(path, standard_deviation, kernel_size, bound):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    imgBlur_CV2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    laplas_filter = [[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]]

    # laplas_filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    laplassian_img = utils.convolution(imgBlur_CV2, laplas_filter)
    laplassian_img = np.absolute(laplassian_img)
    max_diff = np.max(laplassian_img)
    laplassian_img /= max_diff # нормируем

    img_border = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.any(laplassian_img[i][j] >= bound):
                img_border[i][j] = 255
            else:
                img_border[i][j] = 0

    cv2.imshow(f'laplas_dev{standard_deviation}_ker{kernel_size}_bound-{bound}.jpg', img_border)
import numpy as np





stand = [5,10,100]
ker = [15,3,5]
lower = [[0.1,0.7],[0.3,0.8],[0.4,0.9]]

for b in lower:
    laplassian_method('dataset/test1.jpg', 100, 5, b)
cv2.waitKey(0)
cv2.destroyAllWindows()