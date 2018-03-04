import argparse
import json
import time
from threading import Thread

import cv2
import numpy as np
from pyledstrip import LedStrip


def print_properties(vc):
    print('FRAME_WIDTH: %s' % vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('FRAME_HEIGHT: %s' % vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('FPS: %s' % vc.get(cv2.CAP_PROP_FPS))
    print('FOURCC: %s' % vc.get(cv2.CAP_PROP_FOURCC))
    print('FORMAT: %s' % vc.get(cv2.CAP_PROP_FORMAT))
    print('MODE: %s' % vc.get(cv2.CAP_PROP_MODE))
    print('BRIGHTNESS: %s' % vc.get(cv2.CAP_PROP_BRIGHTNESS))
    print('CONTRAST: %s' % vc.get(cv2.CAP_PROP_CONTRAST))
    print('SATURATION: %s' % vc.get(cv2.CAP_PROP_SATURATION))
    print('HUE: %s' % vc.get(cv2.CAP_PROP_HUE))
    # print('GAIN: %s' % cap.get(cv2.CAP_PROP_GAIN))
    print('EXPOSURE: %s' % vc.get(cv2.CAP_PROP_EXPOSURE))
    print('CONVERT_RGB: %s' % vc.get(cv2.CAP_PROP_CONVERT_RGB))


class VideoStream:
    def __init__(self, src=0):
        # noinspection PyArgumentList
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise Exception('Could not open video')

        self.stream.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        self.stream.set(cv2.CAP_PROP_CONTRAST, 0.5)
        self.stream.set(cv2.CAP_PROP_SATURATION, 0.5)
        # vc.set(cv2.CAP_PROP_EXPOSURE, -8.0)
        print_properties(self.stream)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.thread = None

    def start(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        height, width = self.frame.shape[:2]
        return self.grabbed, cv2.resize(
            cv2.GaussianBlur(self.frame, (5, 5), 0),
            (int(width / 2), int(height / 2)),
            interpolation=cv2.INTER_AREA
        )

    def stop(self):
        self.stopped = True
        self.thread.join()

    def finish(self):
        self.stream.release()


def gray_code_to_decimal(gray_code):
    bin_str = gray_code[0]

    for cnt in range(len(gray_code) - 1):
        bin_str += '1' if bin_str[cnt] != gray_code[cnt + 1] else '0'

    return int(bin_str, 2)


class Pixel:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.frame_value = []
        self.contrast = 0
        self.errors = 0
        self.index = -1

    def add_value(self, v):
        self.frame_value.append(v)

    def evaluate(self):
        min_value = float(np.min(self.frame_value))
        max_value = float(np.max(self.frame_value))
        self.contrast = max(max_value - min_value, 1)
        frame_value_norm = [(float(v) - min_value) / self.contrast for v in self.frame_value]
        gray_code = ''

        for i in range(0, len(frame_value_norm), 2):
            first = frame_value_norm[i] > 0.5
            second = frame_value_norm[i + 1] > 0.5

            if first == second:
                self.errors += 1

            gray_code += '1' if first else '0'

        self.index = gray_code_to_decimal(gray_code)


def analyze(frames):
    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    pixels = [[Pixel(x, y) for x in range(width)] for y in range(height)]

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for y in range(height):
            for x in range(width):
                pixels[y][x].add_value(frame[y][x])

    for y in range(height):
        for x in range(width):
            pixels[y][x].evaluate()

    return pixels


def draw_pixels(pixels, led_count):
    width = len(pixels[0])
    height = len(pixels)
    image = np.zeros((height, width, 3), np.float32)
    for y in range(height):
        for x in range(width):
            pixel = pixels[y][x]
            valid = pixel.index < led_count
            filtered = pixel.errors > 0 or pixel.contrast < 30
            image[y][x][0] = 360.0 * pixel.index / led_count if valid and not filtered else 0
            image[y][x][1] = 1.0 / (pixel.errors + 1) if valid and not filtered else 0
            image[y][x][2] = pixel.contrast / 255 if valid and not filtered else 0
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imshow('Evaluation', cv2.resize(image, (int(4 * width), int(4 * height)), interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_index_position(pixels, i):
    width = len(pixels[0])
    height = len(pixels)
    xs = []
    ys = []

    for y in range(height):
        for x in range(width):
            pixel = pixels[y][x]
            if pixel.index == i:
                if not (pixel.errors > 0 or pixel.contrast < 30):
                    xs.append(pixel.x)
                    ys.append(pixel.y)

    if len(xs) == 0:
        return False, None, None

    return True, np.median(xs), np.median(ys)


def get_index_positions_json(pixels, led_count):
    result = []
    for i in range(led_count):
        success, x, y = get_index_position(pixels, i)
        if success:
            entry = {
                'id': i,
                'x': x,
                'y': y
            }
            result.append(entry)

    return json.dumps(result, indent=2)


def string_to_file(string, file):
    with open(file, 'w') as text_file:
        print(string, file=text_file)


def get_index_positions(pixels, led_count):
    positions = {}
    for i in range(led_count):
        success, x, y = get_index_position(pixels, i)
        if success:
            positions[i] = (x, y)

    return positions


def main(args):
    stream = VideoStream(0)
    stream.start()

    window_name = 'Preview'
    cv2.namedWindow(window_name)

    strip = LedStrip(args=args)
    for l in range(strip.led_count):
        strip.set_rgb(l, 1.0, 1.0, 1.0)
    strip.transmit()
    time.sleep(2.0)

    gray_code = cv2.structured_light.GrayCodePattern_create(strip.led_count, 1).generate()[1]
    frames = []
    f = 0.1

    for i in range(len(gray_code)):
        c = gray_code[i][0]
        for l in range(len(c)):
            strip.set_rgb(l, f * c[l] / 255.0, f * c[l] / 255.0, f * c[l] / 255.0)
        strip.transmit()

        time.sleep(0.5)
        grabbed, frame = stream.read()

        if not grabbed:
            print('Could not read frame')
            exit()

        frames.append(frame)
        cv2.imshow(window_name, frame)
        cv2.waitKey(15)

    print('%d frames recorded' % len(frames))

    strip.off()
    cv2.destroyAllWindows()

    stream.stop()
    stream.finish()

    pixels = analyze(frames)

    positions = get_index_positions(pixels, strip.led_count)
    print(positions)

    ds = time.strftime('%Y-%m-%dT%H:%M:%S')
    fn = 'data/heightmap.%s.json' % ds
    string_to_file(get_index_positions_json(pixels, strip.led_count), fn)
    print('exported heightmap to %s' % fn)

    y_min = min([y for (_, y) in positions.values()])
    y_max = max([y for (_, y) in positions.values()])

    for i, (x, y) in positions.items():
        strip.set_hsv(i, (y - y_min) / (y_max - y_min), 1.0, 1.0)
    strip.transmit()

    draw_pixels(pixels, strip.led_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pyledstrip detector')
    LedStrip.add_arguments(parser)
    main(parser.parse_args())
