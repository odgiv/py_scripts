import numpy as np
import argparse, cv2, math, re, glob, os

"""
This script is for rotating back skewed bounding boxes inside image to straight.
"""
def find_min(vals):
    if len(vals) == 0:
        raise ValueError("input array is empty.")

    min_val = vals[0]

    for val in vals:
        if val < min_val:
            min_val = val
    return min_val


def find_max(vals):
    if len(vals) == 0:
        raise ValueError("input array is empty.")

    max_val = vals[0]

    for val in vals:
        if val > max_val:
            max_val = val
    return max_val


def find_distance_between(point1, point2):
    """
    point is (x, y)
    """ 
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def find_scale(center, image_shape):
    """
    Find max distance between center and image edges
    Multiply it by 2 because 2*max_dist would be length of sides of our new shape
    Return scales that are new_length / old shape lengths for height and width
    """
    dist1 = find_distance_between(center, (0, 0))
    dist2 = find_distance_between(center, (0, image_shape[1]))
    dist3 = find_distance_between(center, (image_shape[0], 0))
    dist4 = find_distance_between(center, (image_shape[0], image_shape[1]))
    max_dist = find_max([dist1, dist2, dist3, dist4])
    return (max_dist * 2 / image_shape[0], max_dist * 2 / image_shape[1])


def fix_coordinate(x, y, img_shape):
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    ## There was an error with this code. Without, it works just fine.
    # if x > img_shape[0]:
    #     x = img_shape[0]        
    # if y > img_shape[1]:
    #     y = img_shape[1]
    
    return x, y



def deskew(image, coordinates):
    """
    x0,y0 .---------. x1,y1
          |         |
          |         |
    x3,y3 .---------. x2,y2
    """

    x0, y0, x1, y1, x2, y2, x3, y3 = coordinates[0], coordinates[1], coordinates[2], coordinates[3], coordinates[4], coordinates[5], coordinates[6], coordinates[7]
    shape = image.shape[:2]

    x0, y0 = fix_coordinate(x0, y0, shape)
    x1, y1 = fix_coordinate(x1, y1, shape)
    x2, y2 = fix_coordinate(x2, y2, shape)
    x3, y3 = fix_coordinate(x3, y3, shape)
    
    try:
        min_x = find_min([x0, x1, x2, x3])
        min_y = find_min([y0, y1, y2, y3])
        max_x = find_max([x0, x1, x2, x3])
        max_y = find_max([y0, y1, y2, y3])
    except ValueError as e:
        print(e)
        return

    center = ((max_x + min_x)/2, (max_y + min_y)/2)  
    left_midpoint = [(x0 + x3)/2, (y0 + y3)/2]

    lenght_top = find_distance_between((x0, y0), (x1, y1))
    lenght_bottom = find_distance_between((x3, y3), (x2, y2))
    lenght_left = find_distance_between((x0, y0), (x3, y3))
    lenght_right = find_distance_between((x1, y1), (x2, y2))

    max_width = find_max([lenght_top, lenght_bottom])
    max_height = find_max([lenght_left, lenght_right])

    """
    tan(angle) = length_of_opposite_side / length_of_adjacent_side
    """
    length_of_adjacent_side = center[0] - left_midpoint[0]
    length_of_opposite_side = center[1] - left_midpoint[1]

    rotation_degree = math.atanh(length_of_opposite_side / abs(length_of_adjacent_side)) * 180 / math.pi
    
    
    """
    We introduce scales here to shape and center in order to prevent from cutting out edges when rotating image.
    """
    scale_x, scale_y = find_scale(center, shape)
    #center = (int(center[0] * scale_x), int(center[1] * scale_y))
    shape = (int(shape[0] * scale_x), int(shape[1] * scale_y))
    matrix = cv2.getRotationMatrix2D(center=center, angle=rotation_degree, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

    bbox_width = int(max_width)
    bbox_height = int(max_height)

    return image, center, (bbox_width, bbox_height)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False)    
    args = vars(ap.parse_args())

    if args["image"]:        
        image_names = [str(args["image"])]
    else:
        image_names = glob.glob("./data/*.jpg") + glob.glob("./data/*.png") + glob.glob("./data/*.JPG") + glob.glob("./data/*.PNG") # for debugging purpose

    for image_name in image_names:
        image_name = os.path.basename(image_name)

        # Image must end with proper formats and not contain texts "crop" or "deskewed".
        if re.search(r'^((?!crop|deskewed).)*(.jpg|.png|.JPG|.PNG)$', image_name):        
            try:
                image = cv2.imread("./data/" + image_name)
            except Exception as e:
                print(e)
                continue
        else:
            continue
        
        try:
            with open("./data/" + image_name[:-4] + ".txt") as f:
                lines = f.readlines()
        except IOError as e:
            print(e)
            continue

        if not lines:
            print("No bounding boxes were read from file.")
            exit(0)

        for i, line in enumerate(lines):        
            coordinates =[int(coordinate) for coordinate in line.strip().split(',')]
            new_image, center, (width, height) = deskew(image, coordinates)
            x = int(center[0] - width/2)
            y = int(center[1] - height/2)
            crop = new_image[y:y+height, x:x+width]
            #cv2.imwrite("./data/" + image_name[:-4] + "_deskewed_" + str(i) + image_name[-4:], new_image)
            cv2.imwrite("./data/" + image_name[:-4] + "_crop_" + str(i) + image_name[-4:], crop)
        