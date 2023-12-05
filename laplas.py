import cv2
import numpy as np


def convolution(img, kernel):
    kernel_size = len(kernel)
    # начальные координаты для итераций по пикселям
    x_start = kernel_size // 2
    y_start = kernel_size // 2
    # переопределение матрицы изображения для работы с каждым внутренним пикселем
    matr = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr[i][j] = img[i][j]

    for i in range(x_start, len(matr) - x_start):
        for j in range(y_start, len(matr[i]) - y_start):
            # операция свёртки - каждый пиксель умножается на соответствующий элемент ядра свертки, а затем все произведения суммируются
            val = 0
            for k in range(-(kernel_size // 2), kernel_size // 2 + 1):
                for l in range(-(kernel_size // 2), kernel_size // 2 + 1):
                    val += img[i + k][j + l] * kernel[k + (kernel_size // 2)][l + (kernel_size // 2)]
            matr[i][j] = val
    return matr

def laplassian_method(path,num, standard_deviation, kernel_size, bound):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    imgBlur_CV2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    laplas_filter = [[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]]

    # laplas_filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    laplassian_img = convolution(imgBlur_CV2, laplas_filter)
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

    # cv2.imshow(f'laplas_dev{standard_deviation}_ker{kernel_size}_bound-{bound}.jpg', img_border)
    cv2.imwrite(f'result_pictures/laplas/test{num}_dev{standard_deviation}_ker{kernel_size}_bound-{bound}.jpg',img_border)



stand = [5,10,100]
ker = [3,5,9]
lower = [[0.1,0.7],[0.3,0.8],[0.4,0.9]]

for i in stand:
    for j in ker:
        for l in lower:
            laplassian_method('dataset/test1.jpg',1, i, j, l)
            laplassian_method('dataset/test2.jpg',2, i, j, l)
            laplassian_method('dataset/test3.jpg',3, i, j, l)
            laplassian_method('dataset/test4.jpg',4, i, j, l)
            laplassian_method('dataset/test5.jpg',5, i, j, l)
cv2.waitKey(0)
cv2.destroyAllWindows()