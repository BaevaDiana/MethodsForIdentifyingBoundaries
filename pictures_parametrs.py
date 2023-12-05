import cv2
import os

# путь к папке с изображениями
image_folder = "dataset"

# получаем список изображений в папке
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

# проходимся по каждому изображению
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # загружаем изображение
    image = cv2.imread(image_path)

    #получаем ширину, высоту и количество каналов изображения
    height, width, channels = image.shape

    # выводим полученные параметры
    print(f"Информация об изображении: {image_file}")
    print(f"Ширина: {width} пикселей")
    print(f"Высота: {height} пикселей")
    print(f"Количество каналов: {channels}")
    print("-" * 40)

