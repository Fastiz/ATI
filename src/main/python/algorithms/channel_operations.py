from PIL.Image import Image
import math
import matplotlib.pyplot as plt


def channel_mean(channel: Image):
    h, w = channel.size

    sum = 0

    for x in range(w):
        for y in range(h):
            sum += channel.getpixel((y, x))

    return sum / (h * w)


def channel_variance(channel: Image):
    h, w = channel.size
    mean = channel_mean(channel)

    sum = 0
    for x in range(w):
        for y in range(h):
            sum += (channel.getpixel((y, x)) - mean) ** 2

    return sum / (h * w)


def channel_standard_deviation(channel: Image):
    return math.sqrt(channel_variance(channel))


def channel_contrast(channel: Image, k: int):
    h, w = channel.size
    mean = channel_mean(channel)
    deviation = channel_standard_deviation(channel)

    r1 = mean - deviation
    r2 = mean + deviation

    s1 = r1 / k
    s2 = 255 - (255 - r2) / k

    for x in range(w):
        for y in range(h):
            p = channel.getpixel((y, x))
            if p < r1:
                (x1, x2), (y1, y2) = (0, r1), (0, s1)
            elif r1 <= p <= r2:
                (x1, x2), (y1, y2) = (r1, r2), (s1, s2)
            else:
                (x1, x2), (y1, y2) = (r2, 255), (s2, 255)

            new_p = round((y2 - y1) * (p - x1) / (x2 - x1) + y1)
            channel.putpixel((y, x), new_p)

    return channel


def channel_threshold(channel: Image, u: int):  # Umbralización
    h, w = channel.size
    for x in range(w):
        for y in range(h):
            p = channel.getpixel((y, x))
            if p < u:
                channel.putpixel((y, x), 0)
            else:
                channel.putpixel((y, x), 255)

    return channel


def channel_negative(channel: Image, u: int):
    h, w = channel.size
    for x in range(w):
        for y in range(h):
            p = channel.getpixel((y, x))
            channel.putpixel((y, x), 255 - p)

    return channel


def channel_histogram(channel: Image, display: bool = False):
    data = [0] * 256

    h, w = channel.size
    for x in range(w):
        for y in range(h):
            data[channel.getpixel((y, x))] += 1

    for i in range(len(data)):
        data[i] /= h * w

    if display:
        plt.hist(data, bins=len(data))
        plt.show()

    return data
