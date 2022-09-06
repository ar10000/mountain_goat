import cv2
import numpy as np
import matplotlib.pyplot as plt
import  webcolors
from mountain_goat.grip_detection import get_grips

def getColorBin(img, tl, br):
    # Creates mask over image focus
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[tl[1]:br[1], tl[0]:br[0]] = 255

    # Effecively quantizes the image when the histogram is made.
    # Useful for grouping similar colors.
    binLen = 4
    numBins = int(256 / binLen)
#     plt.imshow(mask)
    #Finds the most common color/in the histogram/for each color channel.
    binColor = map(
        lambda x: np.argmax(
        [cv2.calcHist([img],[x],mask,[numBins],[0,256])])
        ,[0,1,2])
    binColor= list(binColor)
#     ipdb.set_trace()
    fullColor = map(lambda x: x * binLen, binColor)
    fullColor =list(fullColor)
    return fullColor

def findColors_rgb(img, holds):
    # If no keypoints return nothing
    if len(holds)== 0:
        return []

    # # Shift colorspace to HLS
    # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Preallocate space for color array corresponding to keypoints
    colors = np.empty([len(holds),3])
    for i, key in enumerate(holds):
#         print(i)
        br = (int(key[2]), int(key[3]))
        tl = (int(key[0]), int(key[1]))
#         print(br, tl)
        colors[i] = getColorBin(img,tl,br)
    return colors

def draw_grips(img, holds, colors):
    for i, key in enumerate(holds):
#         print(i)
        br = (int(key[2]), int(key[3]))
        tl = (int(key[0]), int(key[1]))
#         print(br, tl)
        cv2.rectangle(img=img,pt1=tl,pt2=br,color=tuple(colors[i]),thickness=2)
    img = img[...,::-1]
# Display the resulting frame
    fig = plt.imshow(img)
    plt.title("Image with grips")
    plt.show()

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name

    return min_colours[min(min_colours.keys())]

def grip_colors(img_path, model_path):
    """takes in an image path and model path
    returns:
    a dictionary of color names and rgb values
    and draws the colors on the image as well
    """
    image = cv2.imread(img_path)
    grips = get_grips(img_path, model_path)
    colors = findColors_rgb(image, grips)
    color_names ={}
    for color in colors:
        name = closest_colour(color)
        color_names[name]= color
    draw_grips(image, grips, colors)
    return color_names

if __name__ == "__main__":
    model_path = '/home/william/code/ar10000/mountain_goat/raw_data/output/model_final.pth'
    image_path ='/home/william/code/ar10000/mountain_goat/Screenshot 2022-08-29 at 12.07.37.png'
    print(grip_colors(img_path=image_path, model_path=model_path))
