import cv2
import numpy as np
import time

def convolution(img, kernel):
    # Размер ядра свертки
    kernel_size = len(kernel)
    # Начальные координаты для итераций по пикселям
    x_start = kernel_size // 2
    y_start = kernel_size // 2
    # Переопределение матрицы изображения для работы с каждым внутренним пикселем
    matr = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr[i][j] = img[i][j]

    # Итерации по внутренним пикселям для операции свертки
    for i in range(x_start, len(matr) - x_start):
        for j in range(y_start, len(matr[i]) - y_start):
            # Операция свёртки - каждый пиксель умножается на соответствующий элемент ядра свертки,
            # а затем все произведения суммируются
            val = 0
            for k in range(-(kernel_size // 2), kernel_size // 2 + 1):
                for l in range(-(kernel_size // 2), kernel_size // 2 + 1):
                    val += img[i + k][j + l] * kernel[k + (kernel_size // 2)][l + (kernel_size // 2)]
            matr[i][j] = val
    return matr

def laplassian_method(path, num, standard_deviation, kernel_size, bound):
    start_time = time.time()
    # Загрузка изображения в оттенках серого
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Применение гауссовского размытия
    imgBlur_CV2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    # Ядро для фильтра Лапласа
    laplas_filter = [[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]]

    #Применение фильтра Лапласа с использованием функции свертки
    laplassian_img = convolution(imgBlur_CV2, laplas_filter)
    #преобразовать все значения в положительные, потому что после операции свертки значения могут быть как положительными, так и отрицательными.
    laplassian_img = np.absolute(laplassian_img)
    #находим макс значение
    max_diff = np.max(laplassian_img)
    laplassian_img /= max_diff  # Нормировка значений-приведения интенсивности пикселей к определенному диапазону, часто от 0 до 1

    # Применение порогового значения для выделения границ
    img_border = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.any(laplassian_img[i][j] >= bound):
                img_border[i][j] = 255
            else:
                img_border[i][j] = 0

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Время выполнения алгоритма для изображения {num}: {execution_time:.4f} секунд")

    # Оценка контраста изображения
    contrast_value = np.std(img_border)
    print(f"Контраст изображения {num}: {contrast_value:.2f}")

    # Сохранение обработанного изображения в файл
    cv2.imwrite(f'result_pictures/laplas/test{num}_dev{standard_deviation}_ker{kernel_size}_bound-{bound}.jpg', img_border)

# Набор значений для экспериментов
stand = [5, 10, 100]
ker = [3, 5, 9]
lower = [[0.1, 0.7], [0.3, 0.8], [0.4, 0.9]]

# Вызов функции для одного изображения с заданными параметрами
laplassian_method('dataset/test1.jpg', 1, 5, 5, [0.1, 0.7])
