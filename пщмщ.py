import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


def calculate_histogram(image):
    hist = [0] * 256
    for row in image:
        for pixel in row:
            hist[pixel] += 1
    return hist


def otsu_threshold(hist, total_pixels):
    sumB = 0
    wB = 0
    maximum = 0
    threshold = 0
    sum1 = sum(i * hist[i] for i in range(256))

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total_pixels - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) * (mB - mF)
        if between > maximum:
            maximum = between
            threshold = t
    return threshold


def remove_salt_pepper(image):
    height, width = image.shape
    filtered = np.copy(image)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            all_neighbors_same = True
            neighbor_value = -1
            for dx, dy in directions:
                nx, ny = i + dx, j + dy
                if 0 <= nx < height and 0 <= ny < width:
                    neighbor = image[nx, ny]
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


def algorithm1(image):
    gray_image = np.mean(image, axis=2).astype(int)
    hist = calculate_histogram(gray_image)
    total_pixels = gray_image.size
    thresh = otsu_threshold(hist, total_pixels)

    binary = (gray_image > thresh).astype(int) * 255

    denoised = remove_salt_pepper(binary)

    height, width = denoised.shape
    labels = np.zeros((height, width), dtype=int)
    current_label = 0
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for i in range(height):
        for j in range(width):
            if labels[i, j] == 0 and denoised[i, j] != 0:
                current_label += 1
                queue = deque()
                queue.append((i, j))
                labels[i, j] = current_label

                while queue:
                    x, y = queue.popleft()
                    color = denoised[x, y]
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width and labels[nx, ny] == 0 and denoised[nx, ny] == color:
                            labels[nx, ny] = current_label
                            queue.append((nx, ny))

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
              range(current_label + 1)]
    result = np.zeros((height, width, 3), dtype=int)
    for i in range(height):
        for j in range(width):
            if labels[i, j] != 0:
                result[i, j] = colors[labels[i, j]]

    return result


def algorithm2(image):
    gray_image = np.mean(image, axis=2).astype(int)
    hist = calculate_histogram(gray_image)

    peaks = []
    for i in range(1, 255):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peaks.append(i)

    height, width = gray_image.shape
    segments = np.full((height, width), -1, dtype=int)
    current_segment = 0

    for peak in peaks:
        threshold = 10
        for i in range(height):
            for j in range(width):
                if segments[i, j] == -1 and abs(gray_image[i, j] - peak) <= threshold:
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        if 0 <= x < height and 0 <= y < width and segments[x, y] == -1:
                            if abs(gray_image[x, y] - peak) <= threshold:
                                segments[x, y] = current_segment
                                stack.append((x - 1, y))
                                stack.append((x + 1, y))
                                stack.append((x, y - 1))
                                stack.append((x, y + 1))
                    current_segment += 1

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(current_segment)]
    result = np.zeros((height, width, 3), dtype=int)
    for i in range(height):
        for j in range(width):
            if segments[i, j] != -1:
                result[i, j] = colors[segments[i, j]]

    return result


if __name__ == "__main__":
    test_image = plt.imread('комп.JPG')

    result1 = algorithm1(test_image)
    result2 = algorithm2(test_image)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(result1)
    plt.title('Алгоритм 1')
    plt.subplot(1, 2, 2)
    plt.imshow(result2)
    plt.title('Алгоритм 2')
    plt.savefig("image_комп.png")
    plt.show()
    plt.close()