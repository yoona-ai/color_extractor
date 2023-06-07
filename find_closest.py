import pickle
from scipy.spatial import distance
import pandas as pd
from PIL import ImageCms, Image

# Load in the rgb-pantone mapping
rgb2pantone = pickle.load(open("rgb_to_pantone.p", "rb"))

# The following method could be vectorized with something like 
# min(sqrt (rgb_keys - rgb_target .^2))

def rgb_to_pantone(r, g, b):
    rgb_keys = list(rgb2pantone.keys())
    # maximum rgb distance = sqrt(3*(255^2)) = 441.67...
    min_dist = 442
    x1 = (r, g, b)
    i_min = -1
    # iterate pantone color keys
    for i in range(0, len(rgb_keys)):
        # convert the key to a tuple
        x2 = rgb_keys[i].split(", ")
        x2 = tuple(map(int, x2))
        dist = distance.euclidean(x1, x2)
        # update min
        if dist < min_dist:
            min_dist = dist
            i_min = i
    return rgb2pantone[rgb_keys[i_min]]

def rgb_to_cmyk(r, g, b):
    # Normalizing RGB values
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    # Calculating CMYK values
    k = 1 - max(r_norm, g_norm, b_norm)
    c = (1 - r_norm - k) / (1 - k) if (1 - k) != 0 else 0
    m = (1 - g_norm - k) / (1 - k) if (1 - k) != 0 else 0
    y = (1 - b_norm - k) / (1 - k) if (1 - k) != 0 else 0

    # Scaling CMYK values to range [0, 100]
    c = int(c * 100)
    m = int(m * 100)
    y = int(y * 100)
    k = int(k * 100)

    return c, m, y, k
