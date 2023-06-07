"""
Title: Image Processing Script with Input as Command-Line Argument

Description:
In this color-extractor version, th images provided as command-line arguments ("python script.py -i image.jpg")
"""

import math
import os 
import cv2
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from skimage import color
import argparse

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


simple_colours = (
    (0, 0, 0, "black"),
    (165, 42, 42, "brown"),
    (245, 245, 220  , "beige"),
    (128, 128, 128, "grey"),
    (255, 255, 255, "white"),
    (0, 0, 255, "blue"),
    (0, 95, 106, "petrol"),
    (64, 224, 208, "turquoise"),
    (0, 255, 0, "green"),
    (128, 128, 0, "olive"),
    (255, 255, 0, "yellow"),
    (255, 165, 0, "orange"),
    (255, 0, 0, "red"),
    (139, 0, 0, "red"),#dark red
    (255, 192, 203, "pink"),
    (255, 162, 200, "lilac"),
    (192, 192, 192, "silver"),
)

def delta_e(color1_lab, color2):
    color2_rgb = sRGBColor(color2[0] / 255, color2[1] / 255, color2[2] / 255)
    color2_lab = convert_color(color2_rgb, LabColor)
    return delta_e_cie2000(color1_lab, color2_lab)

def nearest_colour(color):
    r, g, b = color
    # Red Color
    color1_rgb = sRGBColor(r / 255, g / 255, b / 255)
    color1_lab = convert_color(color1_rgb, LabColor);

    return \
        min(simple_colours, key = lambda simple_colour: delta_e(color1_lab, simple_colour))[3]

class ColorPaletteExtractor:
    num_clusters = None
    image = None
    color_mode = None
    agglomerative_clustering = False
    histogram = None
    shape = None
    colors = None
    labels = None
    low_cutoff = None
    high_cutoff = None

    # initializes the object with the input image, number of clusters, and cutoff values for filtering pixels
    def __init__(self, image, clusters=5, low=0.0, high=1.0, color_mode='RGB', agglomerative_clustering=False):
        self.num_clusters = clusters
        self.low_cutoff = low
        self.high_cutoff = high
        self.image = image
        self.shape = self.image.shape
        self.color_mode = color_mode
        self.agglomerative_clustering = agglomerative_clustering

    # resizes the input image to a 1-dimensional array of RGB values and normalizes them
    @staticmethod
    def preprocess_image(image):
        return image.reshape((-1, 3)).astype("float32") / 255

    # converts the scaled RGB values back to their original scale (0-255)
    @staticmethod
    def unscale(colors):
        return (colors * 255).astype("uint8").tolist()

    @staticmethod
    def rgb2lab(pixels, back=False):
        if not back:
            return color.rgb2lab([pixels])[0]
        else:
            return color.lab2rgb([pixels])[0]

    # convert RGB values to their respective color spaces
    @staticmethod
    def rgb2hsv(pixels, back=False):
        if not back:
            return color.rgb2hsv(pixels)
        else:
            return color.hsv2rgb(pixels)

    # filters out pixels that have a mean value outside of the given range
    def filter_pixels(self, pixels):
        pixels_mean = pixels.mean(1)
        mask = (self.low_cutoff <= pixels_mean) & (pixels_mean <= self.high_cutoff)
        idx = np.arange(len(pixels))
        return pixels[idx[mask]]

    # performs clustering on the filtered pixels and extracts the dominant colors in the image
    def extract_colors(self):
        img = self.image
        pixels = self.preprocess_image(img)
        filtered_pixels = self.filter_pixels(pixels)

        color_mode_switcher = {
            'RGB': None,
            'LAB': ColorPaletteExtractor.rgb2lab,
            'HSV': ColorPaletteExtractor.rgb2hsv
        }

        adjust_colors = color_mode_switcher.get(self.color_mode, None)
        if adjust_colors:
            filtered_pixels = adjust_colors(filtered_pixels)

        if self.agglomerative_clustering:
            cluster_algorithm = AgglomerativeClustering(n_clusters=self.num_clusters)
        else:
            cluster_algorithm = KMeans(n_clusters=self.num_clusters)
        cluster_algorithm.fit(filtered_pixels)

        if adjust_colors:
            centers = adjust_colors(cluster_algorithm.cluster_centers_, True)
        else:
            centers = cluster_algorithm.cluster_centers_

        self.labels = cluster_algorithm.labels_
        histogram = self.centroid_histogram()  # computes the histogram of cluster labels
        centers = centers[(-histogram).argsort()]
        self.histogram = histogram[(-histogram).argsort()]
        centers = self.unscale(centers)

        # rgb2hex - converts RGB values to hexadecimal color codes
        self.colors = tuple((self.rgb2hex(*color), color, nearest_colour(color), amount) for color, amount in
                            zip(centers, self.histogram))

        return self.colors

    # prints the URLs of websites where the user can see the colors in the palette
    def print_clusters(self):
        print("https://coolors.co/" + "-".join(self.rgb2hex(*c[1]) for c in colors))
        for c in colors:
            chex = self.rgb2hex(*c[1])
            print("https://encycolorpedia.de/" + str(chex))

    # plots the histogram of the cluster labels
    def plot_histogram(self):
        hist = self.histogram

        # appending frequencies to cluster centers
        colors = self.colors

        # creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        # creating color rectangles
        for i in range(self.num_clusters):
            print(hist[i])

            end = start + hist[i] * 500

            # getting rgb values
            r = colors[i][1][0]
            g = colors[i][1][1]
            b = colors[i][1][2]

            # using cv2.rectangle to plot colors
            chart1 = cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)

            start = end

        # display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart1)
        plt.show()

    @staticmethod
    def rgb2hex(r, g, b):
        return f'{r:02x}{g:02x}{b:02x}'

    def recolor_pixels(self):
        h = self.shape[0]
        w = self.shape[1]
        img = np.zeros((w * h, 3))
        labels = self.labels

        for i, color in enumerate(self.colors):
            indices = np.where(labels == i)[0]

            for index in indices:
                img[index] = color[1]

        return img.reshape((h, w, 3)).astype(int)

    def centroid_histogram(self):
        numLabels = np.arange(0, len(np.unique(self.labels)) + 1)
        (hist, _) = np.histogram(self.labels, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        return hist
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-lc", "--lowcutoff", default=0.0, type=float, help="low cut off =< 0.0 (default)")
    ap.add_argument("-hc", "--highcutoff", default=1.0, type=float, help="high cut off >= 1.0 (default)")
    ap.add_argument("-i", "--image", required=True, help="image to use")
    ap.add_argument("-cm", "--colormode", default="RGB", type=str, help="RGB, LAB or HSV")
    ap.add_argument("-a", "--agglomerative", default=False, type=bool,
                    help="use Agglomerative Clustering instead of KMeans")
    ap.add_argument("-c", "--clusters", default=8, type=int, help="amount of cluster - 8 (default)")

    args = vars(ap.parse_args())

    low_cutoff = args["lowcutoff"]
    high_cutoff = args["highcutoff"]
    color_mode = args["colormode"]
    use_agglomerative = args["agglomerative"]
    img_path = args["image"]
    clusters = args["clusters"]

    # open image
    img = cv2.imread(img_path)

    # convert to RGB from BGR
    img_c = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    clusters = [1,2,3,4,5]
    for c in clusters:
        # initialize using constructor
        dc = ColorPaletteExtractor(img_c, c, low_cutoff, high_cutoff, color_mode,
                                   agglomerative_clustering=use_agglomerative)

        # print dominant colors
        colors = dc.extract_colors()
        #print(colors)

        recolored = dc.recolor_pixels()
