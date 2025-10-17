import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random


def halftones(color_image):
    halftone = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)
    for i in range(color_image.shape[0]):
        for j in range(color_image.shape[1]):
            r = color_image[i, j, 0]
            g = color_image[i, j, 1]
            b = color_image[i, j, 2]
            halftone[i, j] = (int(r) + int(g) + int(b)) // 3
    return halftone


def compute_histogram(gray_image):
    hist = [0] * 256
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            value = gray_image[i, j]
            hist[value] += 1
    return hist


def otsu(gray_image):
    hist = compute_histogram(gray_image)
    total_pixels = gray_image.shape[0] * gray_image.shape[1]
    hist_norm = [float(h) / total_pixels for h in hist]

    cum_sum = [0.0] * 256
    cum_mean = [0.0] * 256
    cum_sum[0] = hist_norm[0]
    cum_mean[0] = 0 * hist_norm[0]
    for i in range(1, 256):
        cum_sum[i] = cum_sum[i - 1] + hist_norm[i]
        cum_mean[i] = cum_mean[i - 1] + i * hist_norm[i]

    global_mean = cum_mean[255]

    max_disp = 0.0
    best_edge = 0
    for t in range(256):
        w1 = cum_sum[t]
        if w1 == 0:
            continue
        w2 = 1.0 - w1
        if w2 == 0:
            break
        mean1 = cum_mean[t] / w1
        mean2 = (global_mean - cum_mean[t]) / w2
        disp = w1 * w2 * (mean1 - mean2) ** 2
        if disp > max_disp:
            max_disp = disp
            best_edge = t
    return best_edge


def binarize(gray_image, edge):
    binary = np.zeros_like(gray_image, dtype=np.uint8)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image[i, j] > edge:
                binary[i, j] = 255
            else:
                binary[i, j] = 0
    return binary


def salt_pepper_filter(binary_image):
    height, width = binary_image.shape
    filtered = np.copy(binary_image)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for i in range(height):
        for j in range(width):
            pixel = binary_image[i, j]
            all_neighbors_same = True
            neighbor_value = -1
            for dx, dy in directions:
                nx, ny = i + dx, j + dy
                if 0 <= nx < height and 0 <= ny < width:
                    neighbor = binary_image[nx, ny]
                    if neighbor_value == -1:
                        neighbor_value = neighbor
                    if neighbor != neighbor_value:
                        all_neighbors_same = False
                        break
                else:
                    all_neighbors_same = False
                    break
            if all_neighbors_same and neighbor_value != -1:
                if pixel == 0 and neighbor_value == 255:
                    filtered[i, j] = 255
                elif pixel == 255 and neighbor_value == 0:
                    filtered[i, j] = 0
    return filtered


# Функция для нахождения кластеров по пикам и минимумам гистограммы
def find_clusters(gray_image):
    hist = compute_histogram(gray_image)
    clusters = []
    min_val = 0

    for i in range(1, 255):  # Ищем локальные минимумы
        if hist[i] < hist[i - 1] and hist[i] < hist[i + 1] and hist[i] > 0:
            clusters.append((min_val, i))
            min_val = i + 1
    clusters.append((min_val, 255))  # Последний кластер до 255

    # Удаляем пустые или слишком маленькие кластеры (менее 1% пикселей)
    total_pixels = gray_image.shape[0] * gray_image.shape[1]
    valid_clusters = []
    for start, end in clusters:
        count = sum(hist[start:end + 1])
        if count > total_pixels / 100:  # Минимум 1% пикселей
            valid_clusters.append((start, end))
    return valid_clusters


# Функция для выбора семян (ближайший пиксель к пику в кластере)
def find_seeds(gray_image, clusters):
    height, width = gray_image.shape
    seeds = []
    hist = compute_histogram(gray_image)

    for start, end in clusters:
        # Находим пик (максимум) в диапазоне
        peak_value = max(hist[start:end + 1])
        peak_index = start
        for i in range(start, end + 1):
            if hist[i] == peak_value:
                peak_index = i
                break

        # Ищем ближайший пиксель к пику
        min_diff = float('inf')
        best_seed = None
        for i in range(height):
            for j in range(width):
                if start <= gray_image[i, j] <= end and gray_image[i, j] != 0:
                    diff = abs(gray_image[i, j] - peak_index)
                    if diff < min_diff:
                        min_diff = diff
                        best_seed = (i, j)
        if best_seed:
            seeds.append(best_seed)
    return seeds


# Обновлённая функция region_growing с гистограммным методом
def region_growing(image, is_binary=False, t_edge=2.0):
    height, width = image.shape
    labels = np.zeros((height, width), dtype=int)
    current_label = 0
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    segment_stats = {}

    # Гистограммный метод: определяем кластеры и семена
    clusters = find_clusters(image)
    seeds = find_seeds(image, clusters)
    if not seeds:  # Если семян нет
        seeds = [(0, 0)]  # Заглушка

    # Выращивание семян
    for seed_i, seed_j in seeds:
        if labels[seed_i, seed_j] == 0 and image[seed_i, seed_j] != 0:
            current_label += 1
            queue = deque()
            queue.append((seed_i, seed_j))
            labels[seed_i, seed_j] = current_label

            if is_binary:
                while queue:
                    x, y = queue.popleft()
                    color = image[x, y]
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width and labels[nx, ny] == 0 and image[nx, ny] == color:
                            labels[nx, ny] = current_label
                            queue.append((nx, ny))
            else:
                val = float(image[seed_i, seed_j])
                segment_stats[current_label] = {'mean': val, 'var': 0.0, 'count': 1}
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width and labels[nx, ny] == 0:
                            y_val = float(image[nx, ny])
                            stats = segment_stats[current_label]
                            N = stats['count']
                            X_old = stats['mean']
                            S2_old = stats['var']
                            diff = y_val - X_old
                            if N == 1:
                                if abs(diff) > 20:
                                    continue
                                T = 0.0
                            else:
                                eps = 1e-8
                                if abs(S2_old) < eps:
                                    if abs(diff) < 1e-6:
                                        T = 0.0
                                    else:
                                        T = float('inf')
                                else:
                                    T_squared = ((N - 1) * N / (N + 1)) * (diff ** 2) / S2_old
                                    T = T_squared ** 0.5
                            if T <= t_edge:
                                labels[nx, ny] = current_label
                                queue.append((nx, ny))

                                X_new = (N * X_old + y_val) / (N + 1)
                                S2_new = S2_old + (y_val - X_old) ** 2 + N * (X_new - X_old) ** 2
                                segment_stats[current_label]['mean'] = X_new
                                segment_stats[current_label]['var'] = S2_new
                                segment_stats[current_label]['count'] = N + 1
    return labels, current_label


def color_segments(labels, num_segments):
    height, width = labels.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    colors = {}
    for label in range(1, num_segments + 1):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors[label] = (r, g, b)
    colors[0] = (0, 0, 0)

    for i in range(height):
        for j in range(width):
            colored[i, j] = colors[labels[i, j]]
    return colored


def algorithm1(gray_image):
    threshold = otsu(gray_image)
    binary = binarize(gray_image, threshold)
    filtered = salt_pepper_filter(binary)
    labels, num_segments = region_growing(filtered, is_binary=True)
    colored = color_segments(labels, num_segments)
    return colored


def algorithm2(gray_image):
    labels, num_segments = region_growing(gray_image, is_binary=False, t_edge=2.0)
    colored = color_segments(labels, num_segments)
    return colored


if __name__ == "__main__":
    image_paths = ['дом.jpg', 'комп.jpg']

    for idx, path in enumerate(image_paths, 1):
        color_img = plt.imread(path)
        gray_img = halftones(color_img)

        seg1 = algorithm1(gray_img)
        plt.imsave(f'seg1_image{idx}.png', seg1)

        seg2 = algorithm2(gray_img)
        plt.imsave(f'seg2_image{idx}.png', seg2)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(color_img)
        axs[0].set_title('Оригинал')
        axs[1].imshow(seg1)
        axs[1].set_title('1-й алгоритм')
        axs[2].imshow(seg2)
        axs[2].set_title('2-й алгоритм')

        plt.legend()
        plt.savefig(f"image{idx}.png")
        plt.show()
        plt.close()
