import json
from math import factorial

import matplotlib.pyplot as plt
import numpy as np


# http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


class Led:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def get_xy(self):
        return np.array([self.x, self.y])

    def to_json(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y
        }

    @staticmethod
    def from_json(json):
        id = json["id"]
        x = json["x"]
        y = json["y"]

        return Led(id, x, y)


class LedWorld:
    def __init__(self):
        self.leds = {}

    def add_led(self, led: Led):
        self.leds[led.id] = led

    def size(self, includeNan=True):
        """
        for LEDs 0..299 this will return 300
        :param includeNan:
        :return:
        """
        if includeNan:
            return max(self.leds.keys()) + 1
        else:
            return len(self.leds)

    def plot(self):
        ixys = self.to_np(False)
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)

        ax.plot(ixys[1, :], ixys[2, :])
        ax.invert_yaxis()

        # for i in range(len(ixys[0,:])):
        #     ax.annotate(ixys[0,i], (ixys[1,i], ixys[2,i]))

        fig.show()
        return fig

    def to_np(self, includeNan=True):
        res = np.empty((3, self.size(includeNan)), float)

        if includeNan:
            res[:] = np.nan
            for led in sorted(self.leds.values(), key=lambda l: l.id):
                res[0, led.id] = led.id
                res[1, led.id] = led.x
                res[2, led.id] = led.y
        else:
            for id, led in enumerate(sorted(self.leds.values(), key=lambda l: l.id)):
                res[0, id] = led.id
                res[1, id] = led.x
                res[2, id] = led.y

        return res

    def fill_missing_leds(self):
        locs = np.arange(self.size())
        xys = self.to_np(False)
        ii = xys[0, :]
        xs = xys[1, :]
        ys = xys[2, :]
        resx = np.interp(locs, ii, xs, period=self.size())
        resy = np.interp(locs, ii, ys, period=self.size())

        for i in range(self.size()):
            if not i in self.leds.keys():
                self.add_led(Led(i, resx[i], resy[i]))

    def emplace_np(self, np):
        for i in range(len(np[0, :])):
            id = np[0, i]
            x = np[1, i]
            y = np[2, i]

            self.leds[id].x = x
            self.leds[id].y = y

    def smoothen(self):
        xys = self.to_np(False)
        ii = xys[0, :]
        xs = xys[1, :]
        ys = xys[2, :]

        xs2 = savitzky_golay(xs, 21, 3)
        ys2 = savitzky_golay(ys, 21, 3)

        xys[1, :] = xs2
        xys[2, :] = ys2

        self.emplace_np(xys)

    def to_json(self):
        res = [led.to_json() for led in sorted(self.leds.values(), key=lambda l: l.id)]
        return json.dumps(res, indent=2)

    def to_json_file(self, filename):
        def string_to_file(string, file):
            with open(file, "w") as text_file:
                print(string, file=text_file)

        string_to_file(self.to_json(), filename)

    @staticmethod
    def from_json_file(filename):
        def string_from_file(filename):
            with open(filename, "r") as text_file:
                return text_file.read()

        world = LedWorld()
        leds = json.loads(string_from_file(filename))
        for led in leds:
            world.add_led(Led.from_json(led))

        return world
