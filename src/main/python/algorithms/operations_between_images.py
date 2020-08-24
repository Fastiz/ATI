from src.main.python.utils.ImageWrapper import ImageWrapper


def product(first_image: ImageWrapper, second_image: ImageWrapper):
    return pixel_to_pixel_operation(first_image, second_image, lambda a, b: (a * b) % 255)


def subtraction(first_image: ImageWrapper, second_image: ImageWrapper):
    return pixel_to_pixel_operation(first_image, second_image, lambda a, b: abs((a-b) % 255))


def addition(first_image: ImageWrapper, second_image: ImageWrapper):
    return pixel_to_pixel_operation(first_image, second_image, lambda a, b: (a + b) % 255)


def scalar_multiplication(first_image: ImageWrapper, scalar: float):
    w, h = first_image.dimensions()

    result = ImageWrapper.from_dimensions(w, h)

    for x in range(w):
        for y in range(h):
            pixel = first_image.get_pixel(x, y)
            val = [int(first_image.get_pixel(x, y)[i] * scalar) % 255 for i in range(len(first_image.get_pixel(x, y)))]
            result.set_pixel(x, y, tuple(val))

    return result


def pixel_to_pixel_operation(first_image: ImageWrapper, second_image: ImageWrapper, operation):
    assert first_image.dimensions() == second_image.dimensions()

    w, h = first_image.dimensions()

    result = ImageWrapper.from_dimensions(w, h)

    for x in range(w):
        for y in range(h):
            val = [operation(first_image.get_pixel(x, y)[i], second_image.get_pixel(x, y)[i])
                   for i in range(len(first_image.get_pixel(x, y)))]

            result.set_pixel(x, y, tuple(val))

    return result
